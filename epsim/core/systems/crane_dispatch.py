import esper
from ..componets import Idle
from .base_sys import SysBase
from epsim.core import *

class SysDispatch(SysBase):
    def process(self):
        self.go_work_slot()
        self.go_free_slot()
 

    #为处理即将结束的电镀槽指派天车
    def go_work_slot(self):
        ws=self.slot_mgr.get_urgents()
        if len(ws)<1:
            return 
        ws=sorted(ws, key=lambda x: x[0])
        start=0
        while start<len(ws)-1 and len(ws[start][2].op_code)==1:#优先处理加工槽位
            start+=1
        if start==len(ws)-1:
            start==0
        _,slot_id,slot=ws[start]
        free_agvs=[]
        for crane_id, (crane,_) in self.world.get_components(Crane,Idle): 
            f1=self.crane_mgr.in_range(crane_id,slot_id,True)
            f2=self.crane_mgr.can_goto_target(crane_id,crane,slot.offset)
            # if crane.code=='H2':
            #     print('H2')
            if crane.group==slot.group  and f1 and f2 :

                free_agvs.append((abs(crane.offset-slot.offset),crane_id,crane))
        
        free_agvs=sorted(free_agvs, key=lambda x: x[0])#,reverse=True
        if len(free_agvs)<1:
            #print('无可用天车')
            return 
        _,crane_id,crane=free_agvs[0]
        move_dir=0 if abs(slot.offset-crane.offset)<EPS else (slot.offset-crane.offset)/abs((slot.offset-crane.offset))
        self.world.add_component(slot_id,Locked(crane_id,move_dir,0))

        if abs(slot.offset-crane.offset)>EPS:
            debug(f'dispatch free {crane} to {slot}')
            self.world.add_component(crane_id,MoveTo(slot.offset))
            if self.world.has_component(crane_id,Idle):
                self.world.remove_component(crane_id,Idle)
 
    #为挂载物料的天车寻找可用加工槽位
    def go_free_slot(self):
         for crane_id, (crane,wj) in self.world.get_components(Crane,WithJob):
            if self.is_crane_top(crane) and \
                not  self.world.has_component(crane_id,MoveTo) and \
                not  self.world.has_component(crane_id,Down) and \
                not  self.world.has_component(crane_id,Up):
                #print(agv,self.world.has_component(agv_id,MoveTo))
                job=self.world.component_for_entity(wj.job_id,Job)
                self.go_with_job(crane_id,crane,job)



    def go_with_job(self,crane_id,crane,job):
        info=self.get_op_info(job.proc_code,job.op_index)
        slots=[]
        for s_id, (s,_) in self.world.get_components(Slot,Idle):
            if info.code==s.op_code :
                if self.crane_mgr.can_goto_target(crane_id,crane,s.offset):
                    slots.append((abs(s.offset-crane.offset),s))
        
        if len(slots)<1:return
        slots=sorted(slots, key=lambda x: x[0])
        _,slot=slots[0]

        self.world.add_component(crane_id,MoveTo(slot.offset))
        debug(f'{crane} with job to {slot}')

    #离开不属于自己控制的位置
    def run_away(self):
        for crane_id, (crane,_) in self.world.get_components(Crane,Idle):
            for s_id, (s,_) in self.world.get_components(Slot,Wait):
                if len(s.op_code)==1 :continue
                x1,x2=self.slot_mgr.job_next_pos(s_id)
                if x2>crane.max_offset and abs(s.offset-crane.offset)<=1:
                    mv_offset=crane.offset
                    push_dir=-1 if x2>crane.offset else 1
                    mv_offset+=push_dir#*SAFE_CRANE_DISTANCE
                    if mv_offset<crane.min_offset:
                        mv_offset=crane.min_offset
                    if mv_offset>crane.max_offset:
                        mv_offset=crane.max_offset
                    if mv_offset!=crane.offset and self.can_goto_target(crane_id,crane,mv_offset):
                        self.world.add_component(crane_id,MoveTo(mv_offset))
                        debug(f'push {crane} to {mv_offset}')
                        self.world.remove_component(crane_id,Idle)
                        break
                
    # #避免相撞        
    # def run_away2(self):
    #     for crane_id, (crane,lock) in self.world.get_components(Crane,Locked):
            
    #         # if crane.code=='H3':
    #         #     print(crane)
    #         #dir1=0 if abs(mv.offset-offset1)<EPS else (mv.offset-offset1)/abs(mv.offset-offset1)
    #         for agv_id, agv in self.world.get_component(Crane):
    #             if agv.group!=crane.group or crane_id==agv_id : continue
    #             dir=(agv.offset-crane.offset)/abs(agv.offset-crane.offset)
    #             if abs(agv.offset-crane.offset)<SAFE_CRANE_DISTANCE and dir*lock.move_dir>=0:
    #                 go_to=agv.offset+lock.move_dir*SAFE_CRANE_DISTANCE
    #                 if go_to<agv.min_offset:
    #                     go_to=agv.min_offset
    #                 elif go_to>agv.max_offset:
    #                     go_to=agv.max_offset
                    
    #                 self.world.add_component(agv_id,MoveTo(go_to))
    #                 #print(f'{agv} push {crane} to {go_to}')
    #                 if self.world.has_component(crane_id,Idle):
    #                     self.world.remove_component(crane_id,Idle)
    #                 if self.world.has_component(crane_id,Locked):
    #                     self.world.remove_component(crane_id,Locked)
    



            
        #self.world.remove_component(slot_id,ReqCrane)
        # if abs(s.offset-crane.offset)<EPS:
        #     self.world.add_component(slot_id,Locked(0,0))
        # else:

        # self.world.add_component(crane_id,Locked(move_dir,0))
        #print(f'plan: {crane}-->{s}')







            


        





                


