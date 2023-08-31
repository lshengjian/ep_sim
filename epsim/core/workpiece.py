from .world_object import WorldObj

from .constants import *
from typing import Dict
from .componets import OpLimitData,OperateData
from .shapes import get_workpiece_shape
from .rendering import set_color
class Workpiece(WorldObj):
    def __init__(self, x=0,prouct_code='A'):
        self.timer=0
        self.prouct_code=prouct_code
        self.target_op_limit:OpLimitData=None
        super().__init__(x)
    
    def reset(self):
        super().reset()
        self.target_op_limit=None

    
    def set_next_operate(self,pd:OpLimitData,ops_dict:Dict[int,OperateData]):
        self.target_op_limit=pd
        self.color=ops_dict[pd.op_key].color


    @property
    def image(self):
        img=get_workpiece_shape(self.prouct_code)
        r,g,b=self.color.rgb
        img=set_color(img,r,g,b)
        return img        
        
