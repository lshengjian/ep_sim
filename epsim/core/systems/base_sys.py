from esper import Processor
from typing import  Tuple
from ..componets import *
from epsim.core.consts import *
from epsim.core import *


class SysBase(Processor):
    '''
    对进天车和工件进行加工计时
    '''
    def __init__(self,mgrs:Tuple[JobMgr,SlotMgr,CraneMgr]):
        super().__init__()
        self.job_mgr,self.slot_mgr,self.crane_mgr=mgrs
        self.world=self.job_mgr.world
        

    def process(self):
        pass
    
    def reset(self):
        pass

    def get_op_info(self,proc_code:str,op_index:int):
        return self.job_mgr.get_op_info(proc_code,op_index)

    def is_crane_top(self,crane:Crane)->bool:
        return abs(crane.height-H_TOP)<EPS

    def is_crane_low(self,crane:Crane)->bool:
        return abs(crane.height-H_LOW)<EPS