import esper
from .base_sys import *
from ..componets import *
from epsim.core.consts import *
from epsim.core import EVENT_GAME_OVER


class SysUp(SysBase):

    def process(self):
        for crane_id, (crane,up) in self.world.get_components(Crane,Up):
            up.timer+=1
            crane.height+=crane.speed_up_down
            if up.timer==1:
                self.job_out_slot(crane_id,crane)
            if crane.height>=H_TOP:
                crane.height=H_TOP
            #print(crane.height)  
            if abs(crane.height-H_TOP) <EPS: 
                self.world.remove_component(crane_id,Up)
                if self.world.has_component(crane_id,Locked):
                    self.world.remove_component(crane_id,Locked)
                # todo 如果有物料需要等待滴液结束

 


    def job_out_slot(self,crane_id,crane):
        for s_id, s in self.world.get_component(Slot):
            if abs(s.offset-crane.offset)<EPS:
                if  wj:=self.world.try_component(s_id,WithJob) :
                    #job=self.world.component_for_entity(wj.job_id,Job)
                    self.slot_mgr.remove_job(wj.job_id,s_id)
                    self.crane_mgr.add_job(crane_id,wj.job_id)
                break

                        
                        


        
        

   







