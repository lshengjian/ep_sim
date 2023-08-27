from .world_object import WorldObj

from .constants import *
#from typing import List
from .componets import ProcessData
# from .config import next_operate
# from .slot import Slot
class Workpiece(WorldObj):
    def __init__(self, x=0,prouct_code='A'):
        self.timer=0
        self.prouct_code=prouct_code
        self.target_op:ProcessData=None
        super().__init__(x)
    
    def reset(self):
        super().reset()
        self.target_op=None

    
    def set_next_operate(self,pd:ProcessData=None):
        self.target_op=pd

        
        
