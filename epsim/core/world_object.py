from __future__ import annotations

from typing import  Callable

import numpy as np

from epsim.core.constants import (
    COLOR_TO_IDX,
    Actions,
    DIR_TO_VEC,
    Directions,
    OBJECT_TO_IDX,
)



class WorldObj:

    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str,pos:int=0,state:int=0):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.init_state = self.state=state
        self.carrying=None #for HandCrane
        self.attached=None #for workpiece

        # Initial position of the object
        self.init_x: float = pos

        # Current position of the object
        self._x:float = pos
        self._y:float = 1
        self.reset()

    @property
    def x(self):
        pos=self._x
        if self.attached!=None:
            pos=self.attached.x
        return pos 
    @property
    def y(self):
        pos=self._y
        if self.attached!=None:
            pos=self.attached.y
        return pos     
    def reset(self):
        self._x=self.init_x
        self._y=1
        self.state=self.init_state

    
    def step(self,world)->bool:
        self.state=int(self.carrying!=None)
        return True

    def __str__(self):
        flag='[W]' if self.carrying!=None else ' '
        return f'{flag}({self.x:.1f},{self.y:.1f})|{self.color}|{self.state} '

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.state)

  


     
class Goal(WorldObj):
    def __init__(self,pos:int=0):
        super().__init__("goal", "green",pos)

class Workpiece(WorldObj):
    def __init__(self, color="blue",pos:int=0,state=0):
        super().__init__("workpiece", color,pos,state)

class Start(WorldObj):
    def __init__(self,  color: str ="grey",pos:int=0,state: int = 0):
        super().__init__("start",color,pos,state)
    


class End(WorldObj):
    def __init__(self,  color: str ="grey",pos:int=0,state: int = 0):
        super().__init__("end",color,pos,state)
    
class Exchange(WorldObj):
    def __init__(self,  color: str ="grey",pos:int=0,state: int = 0):
        super().__init__("exchange",color,pos,state)

class Tank(WorldObj):
    def __init__(self,  color: str ="blue",pos:int=0,state: int = 0):
        self.timer:int=0
        super().__init__("tank",color,pos,state)
        


    def reset(self):
        super().reset()
        self.timer=0

    def step(self,world)->bool:
        self.state=0
        if self.carrying!=None:
            self.timer+=1
            self.state=self.carrying.state-self.timer
            return True
        return False


 