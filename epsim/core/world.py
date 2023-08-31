from __future__ import annotations
import logging
import math
from typing import  List,Dict
from collections import defaultdict
from .constants import *
from .componets import *
from .world_object import  *
from .crane import Crane
from .slot import Slot
from .workpiece import Workpiece
from .config import build_config
logger = logging.getLogger(__name__)
'''
自动版规则：
1) A类的全部工件加工完后上B的,其余类推
2) 当上料槽位有空时,顺序把待加工物料放到槽位上
3) 当物料放入下料槽位后,物料加工结束,从系统内消失。

加分规则：
1) 在加工槽位出槽时加工的时间在正常加工时段,加1分
2) 物料正常加工结束,加10分

扣分规则：
1) 在加工槽位出槽时物料加工的时间少于最少或多余最多3秒内  减1分
2) 在加工槽位出槽时物料加工的时间少于最少或多余最多超3秒 减3分
3) 放错槽位 游戏结束
4) 天车相撞 游戏结束
'''
class World:
    def __init__(self,config_directory='demo', max_offset:int=32,isAutoPut=True):
        self.config_directory=config_directory
        self.isAutoPut=isAutoPut
        self.is_over=False
        self.reward=0
        self.score=0
        self.step_count=0
        self.ops_dict:Dict[int,OperateData]=None
        self.max_x: int = max_offset
        self.max_y: int = 2
        self.all_cranes:List[Crane] = []
        self.pos_slots:Dict[int,Slot] = {} #Start,End,Belt,Tank
        self.group_slots:Dict[int,List[Slot]]=defaultdict(list)
        self.group_limits:Dict[int,List[int,int]]={}
        self.group_cranes:Dict[int,List[Crane]]=defaultdict(list)
        self.starts:List[Slot]=[]
        self.ends:List[Slot]=[]
        self.product_procs:Dict[str,List[OpLimitData]]=defaultdict(list)
        self.products=[]
        self.prd_idx=0
        self.load_config()
        #self.reset()
    
    def reset(self,ps:List[str]):
        self.products=ps
        self.load_config()
        self.is_over=False
        self.prd_idx=-1
        self.cur_prd_code=None
        self.step_count=0
        self.score=0
        self.products2starts()
        
    def get_group_bound(self,group=1):
        return self.group_limits[group]
    
    def products2starts(self):
        ps=self.products
        if self.isAutoPut:
            for s in self.starts:
                if not s.locked:
                    if len(ps)>0:
                        wp:Workpiece=Workpiece(s.x,ps[0])
                        self.plan_next(wp)
                        s.put_in(wp)
                        ps.remove(ps[0])
        self.products=ps

    def load_config(self):
        self.ops_dict,slots,cranes,procs=build_config(self.config_directory)
        self.group_limits.clear()
         
        self.all_cranes.clear()
        self.group_cranes.clear()

        self.pos_slots.clear()
        self.group_slots.clear()
                
        self.starts.clear()
        self.ends.clear()
        self.product_procs.clear()


        for rec in procs:
            pd:OpLimitData=rec
            self.product_procs[pd.product_code].append(pd)

        for rec in slots:
            s:SlotData=rec
            g=s.group
            x1,x2=self.group_limits.get(g,[1000,-1000])
            for x in s.offsets:
                if x<x1:
                    x1=x
                if x>x2:
                    x2=x 
                self.group_limits[g]=[x1,x2]   
                slot:Slot=Slot(x,s)
                slot.color=self.ops_dict[slot.cfg.op_key].color
                if s.op_key==1:
                    self.starts.append(slot)
                elif s.op_key==3:
                    self.ends.append(slot)
                self.pos_slots[int(x)]=slot
                self.group_slots[g].append(slot)

        for rec in cranes:
            cfg:CraneData=rec
            crane:Crane=Crane(cfg.offset,cfg)
            self.all_cranes.append(crane)
            self.group_cranes[cfg.group].append(crane)
 

    def mask_one_crane(self,crane_idx:int=0):
        crane:Crane=self.all_cranes[crane_idx]
        mask = np.ones(5, dtype=np.int8)
        if crane.y<1e-4:
            mask[Actions.top]=0
        elif crane.y>(self.max_y-1e-4):
            mask[Actions.bottom]=0
        if 1e-4<crane.y<(self.max_y-1e-4):
            mask[Actions.left]=0
            mask[Actions.right]=0
        
        if crane.x<1e-4:
            mask[Actions.left]=0
        elif crane.x>(self.max_x-1e-4):
            mask[Actions.right]=0
        return mask
    

    def plan_next(self,wp:Workpiece):
        ps=self.product_procs[wp.prouct_code]
        if wp.target_op_limit is None:#第一次规划，放到上料位
            wp.set_next_operate(ps[0],self.ops_dict)
            return
        idx=ps.index(wp.target_op_limit)
        if idx<len(ps)-1:
            wp.set_next_operate(ps[idx+1],self.ops_dict)

    def get_free_slot(self,group:int,wp:Workpiece)->Slot:
        '''
        获取指定区间组的指定工艺的空闲工作槽位
        '''
        slots = self.group_slots[group]
        data=[]
        for s in slots:
            #print(s)
            if s.locked==False and s.cfg.op_key==wp.target_op_limit.op_key and s.carrying==None:
                data.append((abs(wp.x-s.x),s))
        if len(data)<1:
            return None
        data.sort(key=lambda x:x[0])
        data[0][1].locked=True
        return data[0][1]
    
  

    def translate(self,source:Crane|Slot,target:Crane|Slot):
        '''
        在天车和工作槽位间转移物料
        '''  
        if target.carrying!=None:
            logger.info(f'{target} already have {target.carrying}')
            self.is_over=True
            return
        if source in self.starts:
            self.products2starts()
        wp,reward=source.take_out()
        self.reward+=reward
        if type(target) is Slot: 
            if target in self.ends:
                self.reward+=10
                del wp
                return 
            if target.cfg.op_key==2:
                x=target.x+1
                target=self.pos_slots[x]

        target.put_in(wp)
        if type(target) is Crane:
            self.plan_next(wp)
         
        



    def check_collide(self,crane:Crane)->bool:
        '''
        天车移动后进行碰撞检测
        '''  
        collide=False
        pos=self.round(crane.x)
        slot=self.pos_slots.get(pos,None)
        if slot!=None and abs(crane.y-1)<0.1:
            wp:Workpiece=crane.carrying
            if wp==None and crane.tip=='↑' and slot.carrying!=None:
                self.translate(slot,crane)
                
            if wp!=None and crane.tip=='↓'  :
                if wp.target_op_limit.op_key!=slot.cfg.op_key:
                    logger.info(f'{wp.target_op_limit.op_key} not same as {slot.cfg.op_key}')
                    self.is_over=True
                    return
                self.translate(crane,slot)
                
                    
        cranes=self.group_cranes[crane.cfg.group]
        for c in cranes:
            if c==crane:
                continue
            if abs(c.x-crane.x)<2:
                collide=True
                logger.info(f'{c} too close to {crane}')
                break

        return collide
        
    def update(self):
        self.step_count+=1
        if self.is_over:
            return
        self.reward=0
        for c in self.all_cranes:
            c.step()

        slots=self.pos_slots.values()
        for s in slots:
            s.step()
        self.check_cranes()
        self.check_slots()
        self.score+=self.reward


    def check_slots(self):
        slots=self.pos_slots.values()
        for s in slots:
            if s.cfg.op_key<10 or s.carrying is None:
                continue
            op:OpLimitData=s.carrying.target_op_limit
            if s.timer>op.max_time+10:
                self.is_over=True
                logger.info(f'{s} op timeout!')
                break  

    def check_cranes(self):
        for c in self.all_cranes:
            if  self.out_bound(c):
                self.is_over=True
                
                return
            if self.check_collide(c):
                self.is_over=True
                #logger.error(f'{c} collided!')
                return
                
    def out_bound(self,crane:Crane):
        x=crane.x
        y=crane.y
        x1,x2=self.group_limits[crane.cfg.group]

        return x<x1 or y<0 or x>x2 or y>self.max_y
    
    def round(self,x:float)->int:
        return int(x+0.5)
    def put_product(self):
        if self.cur_prd_code is None:
            return
        wp=Workpiece(0,self.cur_prd_code)
        self.plan_next(wp)
        start=self.get_free_slot(1,wp)
        if start is None:
            return
        start.put_in(wp)

    def next_product(self):
        self.cur_prd_code=None if len(self.products)<1 else self.products[0]
        if self.cur_prd_code!=None:
            self.products.remove(self.cur_prd_code)
        


    
    def pprint(self):
        for k,cs in self.group_cranes.items():
            print(f'Group {k}')
            for c in cs:
                print(f'{c}')
        print(f'='*18)
        for k,cs in self.group_slots.items():
            print(f'Group {k}')
            for c in cs:
                print(f'{c}')





   





 
                












 

 