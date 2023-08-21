import esper
from .base_sys import *
from ..componets import *
from epsim.core.consts import *


class SysAutoPick(SysBase):
       
    def process(self):
        for crane_id, (crane,_) in self.world.get_components(Crane,Idle):
            if self.crane_mgr.can_pick_job(crane):
                # if crane.code=='H3':
                #     debug(f'{crane} auto pickup')
                self.world.add_component(crane_id,Up())
                self.world.remove_component(crane_id,Idle)

            



