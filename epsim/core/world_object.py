from __future__ import annotations
from .constants import *
from .componets import Color,State
from epsim.utils.onehost import *
class WorldObj:


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
        return int(pos +0.5)
       
    @property
    def y(self):
        pos=self._y
        if self.attached!=None:
            pos=self.attached.y
        return int(pos +0.5) 
        
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
        return f'({self.x:.1f},{self.y:.1f}){flag} '
    
    def state2data(self):
        ds=self.state
        d1=type2onehot(ds.obj_type,SHARE.OBJ_TYPE_SIZE)
        #print(d1.tolist())
        d2=op_ket2onehots(ds.op_key,SHARE.OP_TYPE1_SIZE,SHARE.OP_TYPE2_SIZE)
        d3=np.zeros(SHARE.PRODUCT_TYPE_SIZE,dtype=np.float32) if ds.product_code=='' else \
            type2onehot(ord(ds.product_code[0])-ord('A')+1,SHARE.PRODUCT_TYPE_SIZE)
        #print(d2.tolist(),' ',d3.tolist())
        d4=np.array([ds.x/SHARE.MAX_X,ds.y/SHARE.MAX_Y])
        d5=np.array([ds.op_duration,ds.op_time])/SHARE.MAX_OP_TIME
        #print(d4.tolist(),' ',d5.tolist())
        return np.concatenate((d1,d2,d3,d4,d5),axis=0)






 