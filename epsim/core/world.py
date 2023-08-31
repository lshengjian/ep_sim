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

class World:
    def __init__(self,config_directory='demo', max_offset:int=32):
        self.config_directory=config_directory
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
        self.group_cranes:Dict[int,List[Crane]]=defaultdict(list)
        self.starts=[]
        self.ends=[]
        self.product_procs:Dict[str,List[OpLimitData]]=defaultdict(list)
        self.products={}
        self.prd_idx=0
        self.reset()
    
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
    

    def attach(self, wp:Workpiece, target:Crane|Slot):
        '''
        把物料挂到天车或工作槽位
        '''        
        if target.carrying is None:
            target.carrying=wp
            if type(target) is Crane:
                self.plan_next(wp)
            wp.attached=target

        else:
            self.is_over=True
            logger.info(f'{target} already have something')

    

    def translate(self,source:Crane|Slot,target:Crane|Slot):
        '''
        在天车和工作槽位间转移物料
        '''  
        if target.carrying!=None:
            logger.info(f'{target} already have {target.carrying}')
            self.is_over=True
            return

        self.attach(source.carrying, target)
        source.carrying=None

    def on_workpiece_in_slot(self,wp:Workpiece,slot:Slot):
        # set reward: 1
        slot.timer=0
        if slot.cfg.op_key==3:
            self.reward+=10
            del wp
            slot.carrying=None
            #self.products[wp.prouct_code]

    def on_workpiece_out_slot(self,wp:Workpiece,tank:Slot):
        # set reward: 1
        tank.timer=0
        if tank.cfg.op_key>9 :
            self.reward+=1 if abs(tank.left_time)<3 else -1

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
                self.on_workpiece_out_slot(crane.carrying,slot)
            if wp!=None and crane.tip=='↓'  :
                
                if wp.target_op_limit.op_key!=slot.cfg.op_key:
                    logger.info(f'{wp.target_op_limit.op_key} not same as {slot.cfg.op_key}')
                    self.is_over=True
                    return
                self.translate(crane,slot)
                self.on_workpiece_in_slot(slot.carrying,slot)
                
                    
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
            if s.left_time<0:
                self.reward += -1
                op:OpLimitData=s.carrying.target_op_limit
                if  s.left_time<-5:
                    self.is_over=True
                    logger.info(f'{s} op timeout!')
                    break  

    def check_cranes(self):
        for c in self.all_cranes:
            if  self.out_bound(c.x,c.y):
                self.is_over=True
                
                return
            if self.check_collide(c):
                self.is_over=True
                #logger.error(f'{c} collided!')
                return
                
    def out_bound(self,x:float,y:float):
        return x<0 or y<0 or x>self.max_x or y>self.max_y
    
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
        self.products[self.cur_prd_code][1]+=1
        self.attach(wp,start)

    def next_product(self):
        codes=[]
        for code,d in self.products.items():
            if d[1]<d[0]:
                codes.append(code)
        self.prd_idx=(self.prd_idx+1)%len(codes)
        self.cur_prd_code=None if len(codes)<1 else codes[self.prd_idx]
        

    def reset(self):
        self.is_over=False
        self.prd_idx=-1
        
        self.step_count=0
        self.score=0
        self.ops_dict,slots,cranes,procs=build_config(self.config_directory)
        self.all_cranes.clear()
        self.group_cranes.clear()

        self.pos_slots.clear()
        self.group_slots.clear()
                
        self.starts.clear()
        self.ends.clear()
        self.product_procs.clear()

        self.cur_prd_code=None

        for rec in procs:
            pd:OpLimitData=rec
            self.product_procs[pd.product_code].append(pd)

        #ops=self._make_ops_dict(ds['1-operates'])
        for rec in slots:
            s:SlotData=rec
            for x in s.offsets:
                slot:Slot=Slot(x,s)
                slot.color=self.ops_dict[slot.cfg.op_key].color
                if s.op_key==1:
                    self.starts.append(slot)
                elif s.op_key==3:
                    self.ends.append(slot)
                self.pos_slots[int(x)]=slot
                for g in s.group:
                    self.group_slots[g].append(slot)

        for rec in cranes:
            cfg:CraneData=rec
            crane:Crane=Crane(cfg.offset,cfg)
            self.all_cranes.append(crane)
            self.group_cranes[cfg.group].append(crane)
        
        for d in self.products.values():
            d[1]=0
    
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





   





 
                












 

 