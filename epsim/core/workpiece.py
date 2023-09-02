from .world_object import WorldObj

from .constants import *
from typing import Dict
from .componets import OpLimitData,OperateData
from .shapes import get_workpiece_shape
from .rendering import set_color
class Workpiece(WorldObj):
    UID={}
    def __init__(self, product_code='A',x=0):
        self.timer=0
        self.product_code=product_code
        self.target_op_limit:OpLimitData=None
        super().__init__(x)
    
    @staticmethod
    def make_new(code='A',x=0):
        rt=Workpiece(code,x)
        rt.inc_uid()
        return rt
    def __str__(self):
        return f'{self.product_code}-{self.id} ({self.x},{self.y})'
        
    def reset(self):
        super().reset()
        self.target_op_limit=None

    def inc_uid(self):
        uid=Workpiece.UID.get(self.product_code,0)
        self.id=uid+1
        Workpiece.UID[self.product_code]=self.id

    def set_next_operate(self,pd:OpLimitData,ops_dict:Dict[int,OperateData]):
        self.target_op_limit=pd
        self.color=ops_dict[pd.op_key].color


    @property
    def image(self):
        img=get_workpiece_shape(self.product_code)
        r,g,b=self.color.rgb
        img=set_color(img,r,g,b)
        return img        
        
