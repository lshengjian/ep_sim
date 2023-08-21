import esper
from esper import Processor
#from my_ecs.utils import get_op_info
from ...core.componets import *


class SysWait(Processor):
    def __init__(self):
        super().__init__()

    def process(self):
        for crane_id, (crane,wt,wj) in self.world.get_components(Crane,Wait,WithJob):
            wt.timer+=1
            if wt.timer>=wt.duration :
                self.world.remove_component(crane_id,Wait)
                print(f'{crane} waiting over')

                #if not self.world.has_component(crane_id,MoveTo): #不是到达时等待晃动停止
                #if abs(crane.height-H_HIGHT)<EPS:
                wj = self.world.component_for_entity(crane_id,WithJob)
                job=self.world.component_for_entity(wj.job_id,Job)
                esper.dispatch_event('req_next',crane_id,crane,job)
