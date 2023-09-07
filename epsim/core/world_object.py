from __future__ import annotations
from .constants import *
from .componets import Color,State
from epsim.utils.onehost import *
class WorldObj:
    TILE_SIZE:int=32
    MAX_X:int=32
    MAX_Y:int=2
    MAX_OP_TIME:int=100
    OBJ_TYPE_SIZE:int=3
    OP_TYPE1_SIZE:int=3
    OP_TYPE2_SIZE:int=6
    PRODUCT_TYPE_SIZE:int=3

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
    def state(self)->State:
        pass
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
        self.carrying=None
        self.attached=None


    
    @property
    def image(self):
        pass

    def step(self):
        pass
    def __str__(self):
        flag='[W]' if self.carrying!=None else '  '
        return f'{flag} ({self.x:.1f},{self.y:.1f})'
    
    def state2data(self):
        ds=self.state
        d1=type2onehot(ds.obj_type,self.OBJ_TYPE_SIZE)
        #print(d1.tolist())
        d2=op_ket2onehots(ds.op_key,self.OP_TYPE1_SIZE,self.OP_TYPE2_SIZE)
        d3=np.zeros(self.PRODUCT_TYPE_SIZE,dtype=np.float32) if ds.product_code=='' else \
            type2onehot(ord(ds.product_code[0])-ord('A')+1,self.PRODUCT_TYPE_SIZE)
        #print(d2.tolist(),' ',d3.tolist())
        d4=np.array([ds.x/self.MAX_X,ds.y/self.MAX_Y])
        d5=np.array([ds.op_duration,ds.op_time])/self.MAX_OP_TIME
        #print(d4.tolist(),' ',d5.tolist())
        return np.concatenate((d1,d2,d3,d4,d5),axis=0)



 
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




 