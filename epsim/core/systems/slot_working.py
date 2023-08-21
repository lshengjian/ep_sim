import esper
from .base_sys import SysBase
from ..componets import *
from epsim.core import *

class SysWorking(SysBase):
    '''
    对进入槽内的物料进行电镀
    '''
    def process(self):
        
        for s_id, (s,w,wj) in self.world.get_components(Slot,Wait,WithJob):
            w.timer+=1
            if job:=self.world.try_component(wj.job_id,Job):
                esper.dispatch_event(EVENT_OP_DOING,job,s)
                #print(s,w.timer)
            if  len(s.op_code)>1 and (w.timer-w.duration)> +max(w.duration*0.1,MIN_WAIT_TIME):
                esper.dispatch_event(EVENT_GAME_OVER,f'{job} time out at {s}',-5)
                return
            # if  w.timer +NOTIFY_BEFORE_FINISH>= w.duration and not self.world.has_component(s_id,Locked):
            #     if not self.world.has_component(s_id,ReqCrane):
            #         self.world.add_component(s_id,ReqCrane())

        for s_id, (s,lock) in self.world.get_components(Slot,Locked):
            lock.timer+=1
            if lock.timer>NOTIFY_BEFORE_FINISH:
                self.world.remove_component(s_id,Locked)
                debug(f'remove lock for {s}')

                
                


  



            
