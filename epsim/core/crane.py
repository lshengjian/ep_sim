from .world_object import WorldObj
from .constants import *

class Crane(WorldObj):
    def __init__(self,  color: str ="white",pos:int=0,state: int = 0):
        self.timer:int=0
        self.action:Actions=Actions.stay
        self.tip=''
        super().__init__("crane",color,pos,state)
        
    def __str__(self):
        rt=self.tip+' '+super().__str__()
        return rt

    def reset(self):
        super().reset()
        self._y=2.0
        self.timer=0
        self.action=Actions.stay

    def set_command(self,act:Actions):
        self.action=act
        
        self.tip=Directions[self.action]

    def step(self,world,reset_action=False)->bool:
        dir=DIR_TO_VEC[self.action]
        if self.action==Actions.stay:
            return True

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
         
      