from .world_object import WorldObj
from .constants import *
from .componets import CraneData

class Crane(WorldObj):
    def __init__(self,  x:int,cfg:CraneData):
        self.cfg:CraneData=cfg
        self.timer:int=0
        self.action:Actions=Actions.stay
        self.tip=''
        super().__init__(x)
        
    def __str__(self):
        return self.tip+f' {self.cfg.name}'+super().__str__()

    def reset(self):
        super().reset()
        self._y=2.0
        self.timer=0
        self.action=Actions.stay

    def set_command(self,act:Actions):
        self.action=act
        self.tip=Directions[self.action]

    def step(self,reset_action=False):
        if self.action==Actions.stay:
            return
        dir=DIR_TO_VEC[self.action]
        self._x=self.x+dir[0]*self.cfg.speed_x
        self._y=self.y+dir[1]*self.cfg.speed_y
        if reset_action:
            self.action=Actions.stay

        '''
        x2:float=self.x+dir[0]
        y2:float=self.y+dir[1]
        
        if not world.on_bound(x2,y2):
            UserWarning(f'({x2},{y2} is out bound!')
            if reset_action:
                self.action=Actions.stay
            return False

        self._x=x2
        self._y=y2
        rt= world.collide_check(self)
        if reset_action:
            self.action=Actions.stay
        return rt
        
        '''

         
      