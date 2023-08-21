import esper

from .base_sys import SysBase
from epsim.core.consts import *
from epsim.core.componets import *
from epsim.core import EVENT_GAME_OVER
class SysTranslate(SysBase):
    def __init__(self,game):
        super().__init__(game)
    #    esper.set_handler(EVENT_GAME_OVER, self.reset)
        

    def reset(self,msg='',r=0):
        self.ts=[]
        for s_id, s in self.world.get_component(Slot):
            if 'T'==s.op_code:
                self.ts.append(s_id)
        #print(len(self.ts))
                  
    def process(self):
        for s_id, (s,_,wj) in self.world.get_components(Slot,Wait,WithJob):
            if s_id not in self.ts:continue
            idx=self.ts.index(s_id)
            
            if idx>=0 and idx%2==0:
                self.world.remove_component(s_id,Wait)
                self.world.remove_component(s_id,WithJob)
                self.world.add_component(s_id,Idle())
                if self.world.has_component(s_id,Locked):
                    self.world.remove_component(s_id,Locked)
            
                next_id=self.ts[idx+1]
                if self.world.has_component(next_id,Idle):
                    self.world.remove_component(next_id,Idle)
                job=self.world.component_for_entity(wj.job_id,Job)
                
                info=self.get_op_info(job.proc_code,job.op_index)
                self.world.add_component(next_id,Wait(info.op_time))
                self.world.add_component(next_id,WithJob(wj.job_id))
                if self.world.has_component(next_id,Locked):
                    self.world.remove_component(next_id,Locked)
                # if s.code=='T1':
                #     print(info)



        





                


