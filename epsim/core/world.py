from __future__ import annotations

import math
from typing import  List,Dict
from collections import defaultdict
from .constants import *
from .componets import *
from .world_object import  *
from .constants import CraneAction
from .crane import Crane
from .slot import Slot
from .workpiece import Workpiece
from .dispatch import CraneDispatch
from .config import build_config
from epsim.utils import get_state,get_observation # for image
import logging
logger = logging.getLogger(__name__.split('.')[-1])
'''
外部接口
1) 加入一批物料
add_jobs(codes:List[str]) #add_jobs(['A','A','B'])
2) 选择下一个天车为当前控制对象
next_crane()
3) 给当前天车发作业指令
set_command(action:int]) #set_command(0)
4) 给所有天车发作业指令
set_commands(actions:List[int]) #set_command([0,1])
5) 更新系统
update()
6) 获取当前天车的状态
get_observation()
7) 获取全部天车的状态
get_observations()

自动版规则：
1) 当上料槽位有空时,顺序把待加工物料放到槽位上
2) 当物料放入下料槽位后,物料加工结束,从系统内消失。

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
    def __init__(self,config_directory='test01',max_steps=3000,auto_put_starts=False,auto_dispatch_crane=False):
        
        self.config_directory=config_directory
        self.auto_put_starts=auto_put_starts
        self.auto_dispatch_crane=auto_dispatch_crane
        self.is_over=False
        self.todo_cnt=0
        self._rewards={} #key : DP;H11;H12...
        #self._reward=0
        self.is_timeout=False
        self.score=0
        self.step_count=0
        self.ops_dict:Dict[int,OperateData]=None
        
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
        self.cur_crane_index=0
        self.state=None
        self.max_steps=max_steps
        self.masks={}
        self.load_config()
        self.dispatch:CraneDispatch=CraneDispatch(self)

    @property
    def  reward(self):
        return sum(self._rewards.values())

    @property
    def  cur_crane(self):
        return self.all_cranes[self.cur_crane_index]

    def add_jobs(self,ps:List[str]=[]):
        self.products.extend(ps)
        self.todo_cnt+=len(ps)
        if len(self.products)>0:
            self.cur_prd_code=self.products[0]
        #print('add_jobs')
        if self.auto_put_starts:
            self.products2starts()

    def next_crane(self):
        for c in self.all_cranes:
            c.color=Color(255,255,255)
        self.cur_crane_index=(self.cur_crane_index+1)%len(self.all_cranes)
        self.cur_crane.color=Color(255,0,0)
        
    def put_product(self):
        if self.cur_prd_code is None:
            self._rewards[SHARE.DISPATCH_CODE]=-1
            return
        self.products2starts()


    def shift_product(self):
        self.cur_prd_code=None if len(self.products)<1 else self.products[0]
        
        if self.cur_prd_code is None:
            self._rewards[SHARE.DISPATCH_CODE]=-1
            return 
        buff1=[]
        buff2=[]
        
        for p in self.products:
            if p==self.cur_prd_code:
                buff1.append(p)
            else:
                buff2.append(p)
        buff2.extend(buff1) 
        self.products=buff2
        self.cur_prd_code=buff2[0]
        
    def do_dispatch(self,action:DispatchAction):
        self._rewards[SHARE.DISPATCH_CODE]=0
        for i,crane in enumerate(self.all_cranes):
            self._rewards[crane.cfg.name]=0
        if action==DispatchAction.NEXT_PRODUCT_TYPE:
            self.shift_product()
        elif action==DispatchAction.SELECT_CUR_PRODUCT:
            self.put_product()
        

    def set_crane_commands(self,actions:List[CraneAction]):
        assert len(actions)==len(self.all_cranes)
        for i,crane in enumerate(self.all_cranes):
            self._rewards[crane.cfg.name]=0
            crane.set_command(actions[i])

            
    def update(self):
        if self.is_over or self.is_timeout:
            return
        self.step_count+=1

        if self.step_count>self.max_steps:
            self.is_timeout=True
            return
        
        for crane in self.all_cranes:
            crane.step()

        slots=self.pos_slots.values()
        for slot in slots:
            slot.step()
        self._check_cranes()
        self._check_slots()
        if self.todo_cnt<1:
            self.is_over=True
            self._rewards[SHARE.DISPATCH_CODE]+=100
            logger.info("!!!OK!!!")
        elif self.auto_put_starts:
            #print(self.todo_cnt)
            self.products2starts()
        self.score+=self.reward
    
    def num_jobs_in_first_group(self):
        rt=0
        for s in self.group_slots[1]:
            if s.carrying != None:
                rt+=1
        for c in self.group_cranes[1]:
            if c.carrying != None:
                rt+=1
        return rt

    def get_state_img(self,scrern_img:np.ndarray,nrows,ncols): #仿真系统的全部状态数据
        self.state= get_state(scrern_img,nrows,ncols,SHARE.TILE_SIZE)
        return self.state
    
    def get_observation_img(self,agv:Crane):
        return  get_observation(self.state,agv.x,SHARE.MAX_AGENT_SEE_DISTANCE,SHARE.TILE_SIZE,SHARE.MAX_X)

    def get_state(self): #仿真系统的全部状态数据
        rt=[]
        for crane in self.all_cranes:
            rt.append(crane.state2data())
        for slot in self.pos_slots.values():
            rt.append(slot.state2data()) 
        for pcode in self.products:
            wp=Workpiece.make_new(pcode)
            rt.append(wp.state2data())
        for k in range(len(rt),SHARE.MAX_STATE_LIST_LEN):
            rt.append([0.]*len(rt[0]))
        return np.array(rt,dtype=np.float32)
    
    def get_observation(self,agv:Crane):
        group:int=agv.cfg.group
        rt=[]
              
        rt.append(agv.state2data())
        for crane in self.group_cranes[group]:
            if agv==crane:continue
            rt.append(crane.state2data())
        cs=[]
        for slot in self.group_slots[group]:
            dis=abs(slot.x-agv.x)
            if dis<=SHARE.MAX_AGENT_SEE_DISTANCE:
                cs.append((dis,slot))
        if len(cs)>0:
            cs.sort(key=lambda x:x[0])
            size=len(rt)
            for _,slot in cs:
                if size<SHARE.MAX_OBS_LIST_LEN:
                    rt.append(slot.state2data())
                    size+=1
                else:
                    break 
                              
        for k in range(len(rt),SHARE.MAX_OBS_LIST_LEN):
            rt.append([0.]*len(rt[0]))
        rt=np.array(rt,dtype=np.float32)
        #print(rt.shape)
        return rt.ravel()

    def reset(self):
        #self.load_config()
        self.is_over=False
        self.todo_cnt=0
        self.cur_crane_index=0
        self.prd_idx=0
        self.cur_prd_code=None
        self.step_count=0
        self.score=0
        self.products.clear()
        for crane in self.all_cranes:
            crane.reset()
        for slot in  self.pos_slots.values():
            slot.reset()
        Workpiece.UID.clear()
        self.is_timeout=False
        
    def get_crane_bound(self,crane:Crane)->Tuple[Slot|Crane,Slot|Crane]:
        g=crane.cfg.group
        x1,x2=map(int, self.group_limits[g])
        left=None
        right=None
        l_side=list(filter(lambda agv:agv.x<crane.x, self.group_cranes[g]))
        r_side=list(filter(lambda agv:agv.x>crane.x, self.group_cranes[g]))
        l_side.sort(key=lambda c:c.x,reverse=True)
        r_side.sort(key=lambda c:c.x)
        if len(l_side)>0:
            left=l_side[0]
            x1=left.x+1+SHARE.MIN_AGENT_SAFE_DISTANCE

        if len(r_side)>0:
            right=r_side[0]
            x2=right.x-1-SHARE.MIN_AGENT_SAFE_DISTANCE
        return x1,x2,left,right


    
    def products2starts(self):
        ps=self.products[:]
        if len(ps)<1:
            return

        have_empty=False
        for s in self.starts:
            if s.carrying is None:
                have_empty=True
                break
        if not have_empty:
            self._rewards[SHARE.DISPATCH_CODE]-=1
            return
        doing_jobs=self.num_jobs_in_first_group()

        if  doing_jobs>int(len(self.group_cranes[1])*1.5+0.5):
            self._rewards[SHARE.DISPATCH_CODE]-=1
            return
        if self.auto_put_starts and np.random.random()<0.99:
            return
        
        for s in self.starts:
            #print(s)
            if  len(ps)>0  and s.carrying is None:
                wp:Workpiece=Workpiece.make_new(ps[0],s.x)
                #print('products2starts')
                self.plan_next(wp)
                s.put_in(wp)
                wp.start_tick=self.step_count
                wp.end_tick=9999 #todo
                t=0
                for limit in self.product_procs[ps[0]]:
                    t+=limit.duration
                wp.total_op_time=t
                ps.remove(ps[0])
                break
        self.products=ps
        self.cur_prd_code=None if len(ps)<1 else ps[0]

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


        for data in procs:
            pd:OpLimitData=data
            self.product_procs[pd.product_code].append(pd)

        for data in slots:
            s:SlotData=data
            g=s.group
            x1,x2=self.group_limits.get(g,[1000,-1000])
            for x in s.offsets:
                if x<x1:
                    x1=x
                if x>x2:
                    x2=x 
                self.group_limits[g]=[int(x1),int(x2)]   
                slot:Slot=Slot(x,s)
                slot.color=self.ops_dict[slot.cfg.op_key].color
                if s.op_key==SHARE.START_KEY:
                    self.starts.append(slot)
                elif s.op_key==SHARE.END_KEY:
                    self.ends.append(slot)
                self.pos_slots[int(x)]=slot
                self.group_slots[g].append(slot)

        for data in cranes:
            cfg:CraneData=data
            crane:Crane=Crane(cfg.offset,cfg)
            self.all_cranes.append(crane)
            self.group_cranes[cfg.group].append(crane)
        self.max_x: int = max(list(self.pos_slots.keys()))
     


    def _get_next_op_limit(self,wp:Workpiece):
        slot:Slot=wp.attached
        assert slot  is not None
        cur=wp.target_op_limit
        ps=self.product_procs[wp.product_code]
        cur_idx=ps.index(wp.target_op_limit)
        rt=None
        if cur_idx<len(ps)-1:
            rt=ps[cur_idx+1]
        return rt

        # 如果下个工序是交换，则一定要靠交换位的天车才能来提取


                
    def mask2str(self,masks):
        flags=[]
        for i,m in enumerate(masks):
            if m: flags.append(Directions[i])
        return ''.join(flags)

    def get_masks(self,crane:Crane):
        self.dispatch.decision()
        return self.masks[crane.cfg.id]



    

    def plan_next(self,wp:Workpiece):
        ps=self.product_procs[wp.product_code]
        #print(ps[0])
        if wp.target_op_limit is None:#第一次规划，放到上料位
            wp.set_next_operate(ps[0],self.ops_dict)
            return
        idx=ps.index(wp.target_op_limit)
        if idx<len(ps)-1:
            wp.set_next_operate(ps[idx+1],self.ops_dict)

    # def get_free_slot(self,group:int,wp:Workpiece)->Slot:
    #     '''
    #     获取指定区间组的指定工艺的空闲工作槽位
    #     '''
    #     slots = self.group_slots[group]
    #     data=[]
    #     for s in slots:
    #         #print(s)
    #         if s.locked==False and s.cfg.op_key==wp.target_op_limit.op_key and s.carrying==None:
    #             data.append((abs(wp.x-s.x),s))
    #     if len(data)<1:
    #         return None
    #     data.sort(key=lambda x:x[0])
    #     data[0][1].locked=True
    #     return data[0][1]
    
  

    def _translate(self,source:Crane|Slot,target:Crane|Slot):
        '''
        在天车和工作槽位间转移物料
        '''  

        wp,reward=source.take_out()
        if type(target) is Slot: 
            if target in self.ends:
                self._rewards[source.cfg.name]+=10
                self.todo_cnt-=1
                del wp
                return 
            if target.cfg.op_key==SHARE.SWAP_KEY:
                x=target.x+1
                target=self.pos_slots[x]
        if target.carrying!=None:
            logger.info(f'{target} already have {target.carrying}')
            self._rewards[source.cfg.name]-=5
            self.is_over=True
            return 

        target.put_in(wp)
        if type(target) is Crane:
            self._rewards[target.cfg.name]=reward
            self.plan_next(wp)

         
        



    def _check_collide(self,crane:Crane)->bool:
        '''
        天车移动后进行碰撞检测
        '''  
        collide=False
        pos=self._round(crane.x)
        slot=self.pos_slots.get(pos,None)
        if slot!=None and abs(crane.y-1)<0.1:
            wp:Workpiece=crane.carrying
            if wp==None and crane.last_action==CraneAction.top and slot.carrying!=None:
                self._translate(slot,crane)
                
            if wp!=None and crane.last_action==CraneAction.bottom  :
                if wp.target_op_limit.op_key!=slot.cfg.op_key:
                    logger.info(f'{wp.target_op_limit.op_key} not same as {slot.cfg.op_key}')
                    self._rewards[crane.cfg.name]-=5
                    self.is_over=True
                    return
                self._translate(crane,slot)
                
                    
        cranes=self.group_cranes[crane.cfg.group]
        for c in cranes:
            if c==crane:
                continue
            if abs(c.x-crane.x)<SHARE.MIN_AGENT_SAFE_DISTANCE:
                collide=True
                logger.error(f'{c} too close to {crane}')
                if c.last_action!=CraneAction.stay:
                    self._rewards[c.cfg.name]-=5
                if crane.last_action!=CraneAction.stay:
                    self._rewards[crane.cfg.name]-=5
                break

        return collide
        



    def _check_slots(self):
        slots=self.pos_slots.values()
        for s in slots:
            if s.cfg.op_key<SHARE.MIN_OP_KEY or s.carrying is None:
                continue
            op:OpLimitData=s.carrying.target_op_limit
            if s.timer>op.max_time+SHARE.LONG_ALARM_TIME:
                self.is_over=True
                logger.error(f'{s} op timeout!')
                agvs=[]
                for agv in self.group_cranes[s.cfg.group]:
                    if (agv.carrying is None):
                        agvs.append((abs(agv.x-s.x),agv))
                
                if len(agvs)>0:
                    agvs.sort(key=lambda x:x[0])
                    agv=agvs[0][1]  #最近的空闲天车失职！
                    self._rewards[agv.cfg.name]-=5
                    

                break  

    def _check_cranes(self):
        for c in self.all_cranes:
            if  self._out_bound(c):
                logger.error(f'{c} out  bount!')
                self._rewards[c.cfg.name]-=5
                self.is_over=True
                
                return
            if self._check_collide(c):
                self.is_over=True
                #logger.error(f'{c} collided!')
                return
                
    def _out_bound(self,crane:Crane):
        x=crane.x
        y=crane.y
        x1,x2=self.group_limits[crane.cfg.group]
        return x<x1 or y<0 or x>x2 or y>self.max_y
    
    def _round(self,x:float)->int:
        return int(x+0.5)

        


    
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





   





 
                












 

 