from .world_object import WorldObj

from .constants import *
#from typing import List
# from .componets import ProcessData
# from .config import next_operate
# from .slot import Slot
class Workpiece(WorldObj):
    def __init__(self, x=0,timer=0):
        self.timer=timer
        self.cur_proc_idx:int=-1
        super().__init__(x)
    
    def reset(self):
        super().reset()
        self.cur_proc_idx=-1
    # def plan_next(self,ps:List[ProcessData]=[]):
    #     owner=self.attached
    #     if owner!=None and type(owner) is Slot:
    #         s:Slot=owner
    #         self.next=next_operate(self.process,self.cur_idx)

        
        
