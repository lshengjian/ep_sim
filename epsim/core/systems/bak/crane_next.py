from ...utils.consts import *
from ...core.componets import *
from ...utils import *
from ..base_sys import SysBase


class SysNext(SysBase):

    def process(self):
        #为上行悬挂物料的天车选择下个电镀槽
        for crane_id, (crane,wj) in self.world.get_components(Crane,WithJob):
            #print(agv.height,H_TOP)
            if self.is_crane_top(crane) and \
                not  self.world.has_component(crane_id,MoveTo):
                #print(agv,self.world.has_component(agv_id,MoveTo))
                job=self.world.component_for_entity(wj.job_id,Job)
                self.go_next(crane_id,crane,job)

    def go_next(self,crane_id,crane,job):
        info=self.get_op_info(job.proc_code,job.op_index)
        # fisrt_agv=None
        # agv_to=0
        # c_id=0
        for s_id, (s,_) in self.world.get_components(Slot,Idle):
            if info.code!=s.op_code or \
                s.group!=crane.group or \
                s.offset<crane.min_offset or \
                s.offset>crane.max_offset  :
                #print(info.code,s.op_code,s.offset,crane.min_offset,crane.max_offset )
                continue

            # if can_go:
            print(f'{crane} will go to {s}')
            self.world.add_component(crane_id,MoveTo(s.offset))
            break 
        
