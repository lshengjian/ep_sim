from .base_sys import *
from ..componets import *
from epsim.core.consts import *


class SysPush(SysBase):
   
    def push(self,crane_id,crane,push_dir):
        mv_offset=crane.offset
        mv_offset+=push_dir*SAFE_CRANE_DISTANCE
        if mv_offset<crane.min_offset:
            mv_offset=crane.min_offset
        if mv_offset>crane.max_offset:
            mv_offset=crane.max_offset
        if  mv_offset!=crane.offset and self.crane_mgr.can_goto_target(crane_id,crane,mv_offset):
            self.world.add_component(crane_id,MoveTo(mv_offset))
            debug(f'push {crane} to {mv_offset}')
            if  self.world.has_component(crane_id,Idle):
                self.world.remove_component(crane_id,Idle)


    def run_away(self,crane_id,crane):
        for s_id, (s,_) in self.world.get_components(Slot,Wait):
            if s.group==crane.group and abs(s.offset-crane.offset)<SAFE_CRANE_DISTANCE:
                
                if lock:=self.world.try_component(s_id,Locked):
                    if lock.lock_id!=crane_id:
                        self.push(crane_id,crane,lock.move_dir)
                        return
                else:
                    x1,x2=self.slot_mgr.job_next_pos(crane.group,s_id)
                    if  x2>crane.max_offset:
                        move_dir=-1 if x2>crane.max_offset else 1
                        self.push(crane_id,crane,move_dir)
                        return


    def process(self):
        skips=set()
        for _, (_,lock) in self.world.get_components(Slot,Locked):
            skips.add(lock.lock_id)
        for crane_id, (crane,_) in self.world.get_components(Crane,Idle):
            if crane_id in skips: continue
            self.run_away(crane_id,crane)
            



