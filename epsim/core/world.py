from __future__ import annotations

import math
from typing import Any, List,Dict
from collections import defaultdict

from epsim.core.world_object import  *



class World:
    """
    Represent a grid and operations on it
    """


    def __init__(self, max_offset:int=12,max_height:float=2.0):
        self.max_offset: int = max_offset
        self.max_height: float = max_height
        self.pos_objs:Dict[int,WorldObj]={} #Start,End,Exchange,Tank

        self.type_objs:Dict[str,List[WorldObj]]=defaultdict(list)
        #self.grid: list[WorldObj | None] = [None] * (self.rows * self.cols)
    def on_bound(self,x:float,y:float):
        return x>=0 and x<=self.max_offset and y>=0 and y<=self.max_height
    
    def round(self,x:float)->int:
        return int(x+0.5)
    
    def build(self,cfg:"DictConfig"):
        self.type_objs.clear()
        self.pos_objs.clear()

    def get_objs(self,type:str='type'):
        return self.pos_objs[type]

    def _build(self):
        self.type_objs.clear()
        self.pos_objs.clear()
        start=Start(pos=1)
        self.add2cache(start)
        end=End(color='green',pos=self.max_offset-1)
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
        # self.pprint()

        # print('='*18)
        self.move(crane,[0,1]) #y=1
        self.move(crane,[0,1]) #y=2
        assert wp.y==2 and  wp.x==1

        # self.pprint()
        # print('='*18)

        self.move(crane,[1,0])
        self.move(crane,[1,0])
        assert wp.y==2 and  wp.x==3
        # self.pprint()
        # 

        self.move(crane,[0,-1])
        assert wp.y==1 and  wp.x==3
        # self.pprint()
        # print('='*18)
        self.move(crane,[0,-1])
        self.pprint()
        
        assert wp.y==1 and  wp.x==3
        assert crane.y==0 and  crane.x==3
        



    def add2cache(self, obj:WorldObj):
        self.type_objs[obj.type].append(obj)
        if obj.type!='workpiece' and obj.type!='crane':
            self.pos_objs[round(obj.x)]=obj



    def move(self, crane:Crane,dir=[1,0])->bool:
        if all(np.array(dir)==0):
            return False
        x2:float=crane.x+dir[0]
        y2:float=crane.y+dir[1]
        
        if not self.on_bound(x2,y2):
            UserWarning(f'({x2},{y2} is out bound!')
            return False

        x1=crane.x
        crane._x=x2
        crane._y=y2
        self.collide_check(crane)
        return True

    def collide_check(self,  crane:Crane):
        #assert obj.carrying!=None and  math.isclose(obj.y,self.max_height)
        #assert obj.carrying==None and  math.isclose(obj.y,0)
        if math.isclose(crane.y,1) and crane.carrying==None:
            pos=int(crane.x+0.5)
            slot=self.pos_objs.get(pos,None)
            if slot!=None:
                self.translate(slot,crane)
                return
        
        if math.isclose(crane.y,1) and crane.carrying!=None:
            pos=int(crane.x+0.5)
            slot=self.pos_objs.get(pos,None)
            if slot!=None:
                self.translate(crane,slot)
                


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




