from epsim.core import *

from typing import  Tuple,List


class CraneMgr:

    def __init__(self,world,slot_mgr:SlotMgr,args):

        self.world=world

        self.job_mgr=slot_mgr.job_mgr
        self.slot_mgr=slot_mgr

        self.START=args['START']

        self.END=args['END']

        self.PROCS=args['PROCS']

        # self.JOBS=args['JOBS']

        self.CRANES=args['CRANES']
       

        self.nb_cranes_first_group = 0

        self.crane_ids=[]

        self.dict_code2id=dict()
    

    def goto(self,code,action):

        if action==0:

            return

        crane_id=self.dict_code2id[code]

        crane=self.world.component_for_entity(crane_id,Crane)

        self.world.add_component(crane_id,MoveTo(action+crane.offset))



    def make_cranes(self):
        

        self.nb_cranes_first_group=0

        self.crane_ids=[]

        self.dict_code2id=dict()

        for info in self.CRANES:

            agvId=self.world.create_entity(Crane(

                group=info.group,
                code=info.code,
                min_offset=info.min_offset,

                max_offset=info.max_offset,
                offset=info.offset,

                stop_wait=info.stop_wait,
                speed=info.speed,

                speed_up_down=info.speed_up_down,

                height=H_LOW

                ),Count(),Idle())

            self.crane_ids.append(agvId)

            self.dict_code2id[info.code]=agvId

            if info.group==1:

                self.nb_cranes_first_group+=1



    def get_crane_job(self,crane_id):

        if wj:=self.world.try_component(crane_id,WithJob):

            return self.world.component_for_entity(wj.job_id,Job)

        return None
    

    def get_crane_by_code(self,code)->Tuple[int,Crane]:

        for crane_id,crane in self.world.get_component(Crane):

            if crane.code==code:

                return (crane_id,crane)

        return (0,None)

      


    def get_crane_jobs(self,first_group=False):

        rt=[]

        start=self.world.component_for_entity(self.slot_mgr.start_id,Slot)

        for ent, (agv,wj) in self.world.get_components(Crane,WithJob):

            if first_group and agv.group!=start.group:continue

            job=self.world.component_for_entity(wj.job_id,Job)

            rt.append((ent,job))

        return rt
    

    def add_job(self,crane_id,job_id):

        self.world.add_component(crane_id,WithJob(job_id))

        if self.world.has_component(crane_id,Idle):

            self.world.remove_component(crane_id,Idle)


    def remove_job(self,crane_id):

        self.world.remove_component(crane_id,WithJob)

        self.world.add_component(crane_id,Idle())


    def can_pick_job(self,crane)->'Job':

        if not abs(crane.height-H_LOW)<EPS or not self.world.has_components(Idle):

            return None

        for s_id, (s,wt) in self.world.get_components(Slot,Wait):

            if s.op_code=='T' and int(s.code[1:])%2==1: #单号的转移车不处理

                continue

            if abs(s.offset-crane.offset)<EPS  and wt.left<=1: #

                # if s.code=='SX1':

                #     print('SX1')

                _,x2= self.slot_mgr.job_next_pos(crane.group,s_id)

                if crane.min_offset<=x2<=crane.max_offset: #确保能移动到下一作业位置

                    job= self.slot_mgr.get_job(s_id)

                    return job

        return None


    def can_drop_job(self,crane,job)->bool:

        if not abs(crane.height-H_TOP)<EPS or not self.world.has_components(WithJob):

            return False

        for _, (s,_) in self.world.get_components(Slot,Idle):

            if  abs(s.offset-crane.offset)<EPS:#crane.group==s.group and

                info=self.job_mgr.get_op_info(job.proc_code,job.op_index)

                if info.code==s.op_code:

                    #print(job,info)

                    return True

        return False


    def in_range(self,crane_id:int,s_id:int,isWorking=True)->bool:

        crane=self.world.component_for_entity(crane_id,Crane)

        slot=self.world.component_for_entity(s_id,Slot)

        offset1=offset2=slot.offset

        # if crane.code=='H2' and slot.code=='SX1':

        #     print('H2')

        if isWorking:

            offset1,offset2=self.slot_mgr.job_next_pos(crane.group,s_id)

        if offset1<crane.min_offset:

            return False

        if offset2>crane.max_offset:

            return False

        return True



    def can_goto_target(self,crane_id:int,crane:Crane,target:int):

        if target<crane.min_offset or target>crane.max_offset:

            return False

        safe=True

        for agv_id, agv in self.world.get_component(Crane):

            if crane_id==agv_id or crane.group!=agv.group: continue

            agv_to=agv.offset

            if mv:=self.world.try_component(agv_id,MoveTo):

                agv_to=mv.offset

            if not is_safe(crane,target,agv,agv_to):

                safe=False

                break

        return safe

    # def get_states(self)->List[job_state]:

    #     rt=[]

    #     for c_id, c in self.world.get_component(Crane):
    #         rt.append(self.get_state(c_id,c))

    #     return rt  

    # def get_state(self,crane_id,crane):

    #     #crane=self.world.component_for_entity(crane_id,Crane)
    #     rt=job_state(*([0]*7))

    #     rt=rt._replace(type=1)
    #     rt=rt._replace(offset=crane.offset)
        

    #     if wj:=self.world.try_component(crane_id,WithJob):

    #         job=self.world.component_for_entity(wj.job_id,Job)

    #         info=self.job_mgr.get_op_info(job.proc_code,job.op_index)

    #         rt=rt._replace(job_num=int(job.code[1:]))

    #         rt=rt._replace(proc_num=int(job.proc_code[1:]))

    #         rt=rt._replace(op_num=job.op_index+1)
    #         rt=rt._replace(plan_op_time=info.op_time)


    #     return rt


