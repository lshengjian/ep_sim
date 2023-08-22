from __future__ import annotations

import math
from typing import Any, List,Dict
from collections import defaultdict

from .world_object import  *
from .crane import Crane


class World:
    """
    Represent a grid and operations on it
    """


    def __init__(self, max_offset:int=12,max_deep:float=2.0):
        self.max_x: int = max_offset
        self.max_y: float = max_deep
        # self.cranes:List[Crane]=[]
        # self.workpieces:List[Workpiece]=[]
        self.pos_slots:Dict[int,WorldObj]={} #Start,End,Exchange,Tank

        self.type_objs:Dict[str,List[WorldObj]]=defaultdict(list)
        #self.grid: list[WorldObj | None] = [None] * (self.rows * self.cols)
    def on_bound(self,x:float,y:float):
        return x>=0 and x<=self.max_x and y>=0 and y<=self.max_y
    
    def round(self,x:float)->int:
        return int(x+0.5)
    
    def add2cache(self, obj:WorldObj):
        self.type_objs[obj.type].append(obj)
        if obj.type!='workpiece'and  obj.type!='crane':    
            self.pos_slots[round(obj.x)]=obj 

    def build(self,cfg:"DictConfig"):
        self.type_objs.clear()
        self.pos_slots.clear()

    def get_objs(self,type:str='type'):
        return self.pos_slots[type]

    def _build(self):
        self.type_objs.clear()
        self.pos_slots.clear()
        start=Start(pos=1)
        self.add2cache(start)
        end=End(color='green',pos=self.max_x-1)
        self.add2cache(end)
        wp=Workpiece(pos=0,state=5)
        self.add2cache(wp)
        tank1=Tank(pos=3)
        self.add2cache(tank1)
        tank2=Tank(pos=4)
        self.add2cache(tank2)
        self.attach(wp,start)
        assert wp.y==1 and  wp.x==1

        crane=Crane(pos=1)
        self.add2cache(crane)
        assert crane.y==2 and  wp.x==1

        crane.set_command(Actions.up)
        self.update()
        #self.pprint()
        assert wp.y==1 and  wp.x==1 and wp.attached==crane
        self.update()
        assert wp.y==0 and  wp.x==1 
        

        crane.set_command(Actions.forward)
        self.update()
        self.update()
        assert wp.y==0 and  wp.x==3


        crane.set_command(Actions.down)
        self.update()
        assert wp.y==1 and  wp.x==3
        # self.pprint()
        # print('='*18) 
        self.update()
        self.pprint()
       
        assert wp.y==1 and  wp.x==3
        assert crane.y==2 and  crane.x==3

        crane.set_command(Actions.stay)
        self.update()
        assert crane.y==2 and  crane.x==3
        print('='*18)
        self.pprint()
        



    def update(self)->bool:
        cranes=self.type_objs['crane']
        for c in cranes:
            c.step(self)
        slots=self.pos_slots.values()
        for s in slots:
            s.step(self)
        return True


    def collide_check(self, crane:Crane)->bool:
        #assert obj.carrying!=None and  math.isclose(obj.y,self.max_height)
        #assert obj.carrying==None and  math.isclose(obj.y,0)
        collide=False
        pos=self.round(crane.x)
        slot=self.pos_slots.get(pos,None)
        if slot!=None and math.isclose(crane.y,1):
            if crane.carrying==None and crane.tip=='↑':
                self.translate(slot,crane)
            elif crane.carrying!=None and crane.tip=='↓':
                self.translate(crane,slot)
        cranes=self.type_objs['crane']
        for c in cranes:
            if c==self:
                continue
            if abs(c.x-crane.x)<2:
                collide=True
                UserWarning(f'{c} too close to {crane}')
                break

        return collide
                


    def attach(self,wp:Workpiece,target:Start):
        if target.carrying is None:
            self.link(wp, target)
        else:
            UserWarning(f'{target} already have something')

    def link(self, wp, target):
        target.carrying=wp
        wp.attached=target

    def translate(self,source:WorldObj,target:WorldObj):
        if target.type=='workpiece':
            UserWarning(f'{target} is workpiece!')
            return
        if target.carrying!=None:
            UserWarning(f'{target} already have something')
            return

        if source.carrying!=None :
            self.link(source.carrying, target)
            source.carrying=None
   
    def pprint(self):
        for k,objs in self.type_objs.items():
            print(k)
            print('   ',end='')
            for obj in objs:
                print(obj,end=' ')
            print('')




