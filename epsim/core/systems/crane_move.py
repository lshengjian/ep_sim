import esper
from .base_sys import *
from ..componets import *
from epsim.core.consts import *
from epsim.core import EVENT_GAME_OVER

class SysMoveTo(SysBase):

    def check_safe(self,crane_id,crane)->bool:
        for agv_id, agv in self.world.get_component(Crane):
            if crane_id==agv_id or agv.group!=crane.group: continue
            df=agv.offset-crane.offset
            if abs(df)<SAFE_CRANE_DISTANCE:
                esper.dispatch_event(EVENT_GAME_OVER,'两天车距离太近了！',-10)
                return False
        return True

    def wait_slot_op(self,crane,move):
        for _, (s,wt) in self.world.get_components(Slot,Wait):
            if abs(crane.offset-s.offset)<EPS and wt.left>0:
                move.wait_time+=1


        
    def process(self):
        for crane_id, (crane,mv) in self.world.get_components(Crane,MoveTo):
            if not self.check_safe(crane_id,crane):
                self.world.remove_component(crane_id,MoveTo)
                if not self.world.has_component(crane_id,WithJob):
                    self.world.add_component(crane_id,Idle())
                return 

            if abs(mv.offset-crane.offset)<EPS:
                mv.wait_time-=1
                #self.wait_slot_op(crane,mv)
                if mv.wait_time<=0:
                    self.world.remove_component(crane_id,MoveTo)
                    if not self.world.has_component(crane_id,WithJob):
                        self.world.add_component(crane_id,Idle())
                    
            else:
                if self.world.has_component(crane_id,Idle):
                    self.world.remove_component(crane_id,Idle)
                dir=(mv.offset-crane.offset)/abs(mv.offset-crane.offset)
                old=crane.offset
                crane.offset+=dir*crane.speed
                #print(crane.offset)
                if arrived_to(mv.offset,old,crane.offset):
                    crane.offset=mv.offset
                job=self.crane_mgr.get_crane_job(crane_id)
                if job:
                    job.offset=crane.offset
                    #job.offset=round(crane.offset)
            



