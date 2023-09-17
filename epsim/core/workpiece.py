from .world_object import WorldObj

from .constants import *
from typing import Dict
from .componets import OpLimitData,OperateData,State
from .shapes import get_workpiece_shape
from .rendering import set_color
class Workpiece(WorldObj):
    UID={}
    def __init__(self, product_code='A',x=0):
        self.timer=0
        self.product_code=product_code
        self.target_op_limit:OpLimitData=None
        self._start_tick=0
        self._end_tick=0
        self._total_op_time=0
        super().__init__(x)

    @property
    def left_time(self):
        return self._end_tick-(self._total_op_time+self._start_tick)
    
    @property
    def end_tick(self):
        return self.end_tick
    
    @end_tick.setter
    def end_tick(self,val):
        self._end_tick=val

    @property
    def start_tick(self):
        return self._start_tick
    
    @start_tick.setter
    def start_tick(self,val):
        self._start_tick=val
    

    @property
    def total_op_time(self):
        return self._total_op_time
   
    @total_op_time.setter
    def total_op_time(self,val):
        self._total_op_time=val
    @property
    def state(self)->State:
        op_limit=self.target_op_limit
        assert op_limit!=None
        return State(ObjType.Workpiece,op_limit.op_key,self.product_code,self.x ,self.y,op_limit.duration)
        
    @staticmethod
    def make_new(code='A',x=0):
        rt=Workpiece(code,x)
        rt.inc_uid()
        return rt
    def __str__(self):
        return f'{self.product_code}-{self.id} ({self.x},{self.y})'
        
    # def reset(self):
    #     super().reset()
    #     self.target_op_limit=None

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
        
