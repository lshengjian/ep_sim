from __future__ import annotations

import math
from typing import Any, List,Dict
from collections import defaultdict
from .constants import *
from .componets import *
from .world_object import  *
from .crane import Crane
from .slot import Slot
from .workpiece import Workpiece
from .config import build_config

default_cfg_dict=build_config()
class World:
    def __init__(self,cfg_dict:Dict[str,List[Index]]=default_cfg_dict, max_offset:int=32,max_deep:float=2.0):
        self.is_over=False
        self.reward=0
        self.max_x: int = max_offset
        self.max_y: float = max_deep
        self.all_cranes:List[Crane] = []
        self.pos_slots:Dict[int,Slot] = {} #Start,End,Belt,Tank
        self.group_slots:Dict[int,List[Slot]]=defaultdict(list)
        self.group_cranes:Dict[int,List[Crane]]=defaultdict(list)
        self.starts=[]
        self.ends=[]
        self.cfg_dict=cfg_dict
        self.product_procs:Dict[str,List[ProcessData]]=defaultdict(list)
        self.build()
    
    def plan_next(self,wp:Workpiece):
        ps=self.product_procs[wp.prouct_code]
        if wp.target_op is None:#第一次规划，放到上料位
            wp.set_next_operate(ps[0])
            return
        idx=ps.index(wp.target_op)
        if idx<len(ps)-1:
            wp.set_next_operate(ps[idx+1])

    def get_free_slot(self,group:int,wp:Workpiece)->Slot:
        '''
        获取指定区间组的指定工艺的空闲工作槽位
        '''
        slots = self.group_slots[group]
        data=[]
        for s in slots:
            if s.locked==False and s.cfg.name==wp.target_op.operate and s.carrying==None:
                data.append((abs(wp.x-s.x),s))
        
        data.sort(key=lambda x:x[0])
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
            print(f'{target} already have something')
    

    def translate(self,source:Crane|Slot,target:Crane|Slot):
        '''
        在天车和工作槽位间转移物料
        '''  
        # if type(target) is Workpiece :
        #     print(f'{target} is workpiece!')
        #     self.is_over=True
        #     return
        if target.carrying!=None:
            print(f'{target} already have {target.carrying}')
            self.is_over=True
            return

        if source.carrying!=None :
            self.attach(source.carrying, target)
            source.carrying=None
    def on_workpiece_out_slot(self,wp:Workpiece,tank:Slot):
        # set reward: 1
        if tank.cfg.name=='下料':
            self.reward+=10
        else:
            self.reward+=1

    def check_collide(self,crane:Crane)->bool:
        '''
        天车移动后进行碰撞检测
        '''  
        collide=False
        pos=self.round(crane.x)
        slot=self.pos_slots.get(pos,None)
        if slot!=None and math.isclose(crane.y,1):
            wp=crane.carrying
            if wp==None and crane.tip=='↑':
                self.translate(slot,crane)
                self.on_workpiece_out_slot(crane.carrying,slot)
            elif wp!=None and crane.tip=='↓':
                # if wp.target_type!=slot.cfg.id:
                #     print(f'{wp.target_type} not same as {slot.type_id}')
                #     self.is_over=True
                #     return
                self.translate(crane,slot)
        cranes=self.group_cranes[crane.cfg.group]
        for c in cranes:
            if c==crane:
                continue
            if abs(c.x-crane.x)<2:
                collide=True
                print(f'{c} too close to {crane}')
                break

        return collide
        
    def update(self):
        for cs in self.group_cranes.values():
            for c in cs:
                c.step()
        slots=self.pos_slots.values()
        for s in slots:
            s.step()

        self.check_cranes()
        self.check_slots()


    def check_slots(self):
        slots=self.pos_slots.values()
        for s in slots:
            if s.left_time<0:
                self.is_over=True
                print(f'{s} op timeout!')
                break  

    def check_cranes(self):
        for c in self.all_cranes:
            if self.check_collide(c):
                self.is_over=True
                print(f'{c} collided!')
                break
                
    def is_in_bound(self,x:float,y:float):
        return x>=0 and x<=self.max_x and y>=0 and y<=self.max_y
    
    def round(self,x:float)->int:
        return int(x+0.5)
  

    def _make_ops_dict(self,ops:List[OperateData]):
        rt={}
        for op in ops:
            rt[op.key]=op
        return rt
    

    def build(self):
        ds=self.cfg_dict
        self.all_cranes.clear()
        self.group_cranes.clear()

        self.pos_slots.clear()
        self.group_slots.clear()
                
        self.starts.clear()
        self.ends.clear()
        self.product_procs.clear()

        for rec in ds['4-procedures']:
            pd:ProcessData=rec
            self.product_procs[pd.product_code].append(pd)

        #ops=self._make_ops_dict(ds['1-operates'])
        for rec in ds['2-slots']:
            s:SlotData=rec
            for x in s.offsets:
                slot:Slot=Slot(x,s)
                if s.name=='上料':
                    self.starts.append(slot)
                elif s.name=='下料':
                    self.ends.append(slot)
                self.pos_slots[int(x)]=slot
                for g in s.group:
                    self.group_slots[g].append(slot)

        for rec in ds['3-cranes']:
            cfg:CraneData=rec
            crane:Crane=Crane(cfg.offset,cfg)
            self.all_cranes.append(crane)
            self.group_cranes[cfg.group].append(crane)
    
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





   





 
                












 

 