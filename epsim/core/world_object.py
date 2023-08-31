from __future__ import annotations
from .constants import *
from .componets import Color
class WorldObj:
    TILE_SIZE:32
    """
    Base class for grid world objects
    """
    def __init__(self,x:int=0):
        self.init_x: float = x
        self.color=Color(255,255,255)

        self.carrying=None #for HandCrane
        self.attached=None #for workpiece

        # Current position of the object
        self._x:float = x
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

    
    @property
    def image(self):
        pass

    def step(self):
        pass
    def __str__(self):
        flag='[W]' if self.carrying!=None else '  '
        return f'{flag} ({self.x:.1f},{self.y:.1f})'

 
'''
class Start(WorldObj):
    def __init__(self, pos:int=0,):
        super().__init__(BUFF_START,"grey",pos)
    

class End(WorldObj):
    def __init__(self,  pos:int=0):
        super().__init__(BUFF_END,"grey",pos)
    
class Belt(WorldObj):
    def __init__(self, pos:int=0):
        super().__init__(BUFF_SWAP,"grey",pos)
'''




 