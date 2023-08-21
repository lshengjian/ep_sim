import esper
from .base_sys import *
from ..componets import *
from epsim.core.consts import *
from epsim.core import EVENT_GAME_OVER

class SysDown(SysBase):

                

    def process(self):
        # for crane_id, (crane,_) in self.world.get_components(Crane,Idle):
        #     if  abs(crane.height-H_TOP)<EPS and \
        #         not self.world.has_component(crane_id,WithJob):
        #         self.world.add_component(crane_id,Down())
        #         self.world.remove_component(crane_id,Idle)
        for crane_id, (crane,_) in self.world.get_components(Crane,Down):
            crane.height-=crane.speed_up_down
            if crane.height<=H_LOW:
                crane.height=H_LOW
            if self.is_crane_low(crane):
                if wj:=self.world.try_component(crane_id,WithJob):
                    for s_id,s in self.world.get_component(Slot):
                        if abs(s.offset-crane.offset)<EPS:
                            self.crane_mgr.remove_job(crane_id)
                            self.slot_mgr.add_job(wj.job_id,s_id)
                            self.world.add_component(crane_id,Idle())
                            self.world.remove_component(crane_id,Down)
                            #self.run_away(crane_id,crane)
                            break


                    
                    

    # def put_job(self,job_id,crane):
    #     job = self.world.component_for_entity(job_id,Job)
    #     info=self.get_op_info(job.proc_code,job.op_index)
    #     found=False
    #     isOver=False
    #     isJobDone=False
    #     for s_id,s in self.world.get_component(Slot):
    #         if abs(s.offset-crane.offset)<EPS:
    #             found=True


    #             if abs(s.offset-self.game.END)<EPS:
    #                 job.end_time=self.game.nb_steps
    #                 esper.dispatch_event('job_finished',job_id,job)
    #                 self.world.add_component(job_id,JobDone)
    #                 #self.world.remove_component(s_id,Idle())
    #                 isJobDone=True
  
    #             elif s.op_code!=info.code:
    #                 esper.dispatch_event('game_over',f'物料放错了槽位:{s}',-5)
    #                 isOver=True
                    
    #             elif self.world.has_component(s_id,WithJob):
    #                 esper.dispatch_event('game_over',f'物料放到了正在加工的槽位:{s}',-5)
    #                 isOver=True
                    
    #             break
    #     if found  :
    #         if not isOver and not isJobDone:
    #             #print(f'put {job} to {s}')
    #             if self.world.has_component(s_id,Idle):
    #                 self.world.remove_component(s_id,Idle)
    #             self.world.add_component(s_id,WithJob(job_id))
    #             self.world.add_component(s_id,Wait(info.op_time))
    #             esper.dispatch_event('op_start',job,s)

    #     else:
    #         esper.dispatch_event('game_over',f'物料放到了地上！',-5)



   


            





