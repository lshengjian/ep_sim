import esper
from .base_sys import *
from ..componets import *
from epsim.core.consts import *

class SysAutoDrop(SysBase):
       
    def process(self):
        for crane_id, (crane,wj) in self.world.get_components(Crane,WithJob):
            job=self.world.component_for_entity(wj.job_id,Job)
            if self.crane_mgr.can_drop_job(crane,job):
                #print(f'{crane} put down job:{job}')
                self.world.add_component(crane_id,Down())

            



