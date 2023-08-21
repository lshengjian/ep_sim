import esper
from ..componets import Count,Idle
from .base_sys import SysBase
from epsim.core.consts import *

class SysCount(SysBase):
    '''
    对进天车和工件进行加工计时
    '''

    def process(self):
        for ent, cnt in self.world.get_component(Count):
            if  free:=self.world.try_components(ent,Idle):
                continue
            cnt.steps+=1
