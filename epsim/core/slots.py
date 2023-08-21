import esper

from typing import Tuple, List
from epsim.core import *


class SlotMgr:

    def __init__(self, world, job_mgr: JobMgr, args):

        self.job_mgr = job_mgr

        # self._args=args

        self.world = world

        self.nb_steps = 0

        self.SLOTS = args['SLOTS']

        self.slot_ids = []

        self.op_types = []

        self.start_id = 0
        self.dict_slot_cranes = dict()

    def step(self):

        self.nb_steps += 1

        self.world.process()

    def get_slot_jobs(self, first_group=False):

        rt = []

        start = self.world.component_for_entity(self.start_id, Slot)

        for ent, (s, wj) in self.world.get_components(Slot, WithJob):

            if first_group and s.group != start.group:
                continue

            job = self.world.component_for_entity(wj.job_id, Job)
            rt.append((ent, job))

        return rt

    def make_slots(self):

        self.slot_ids = []

        self.op_types = []

        self.start_id = 0

        self.nb_steps = 0
        self.dict_slot_cranes = dict()

        for op_code, infos in self.SLOTS.items():

            self.op_types.append(op_code)

            for info in infos:

                s = Slot(info.group, info.code, info.offset, op_code)

                sId = self.world.create_entity(s, Idle())

                self.slot_ids.append(sId)

                if 'S' == op_code:

                    self.start_id = sId

                for agv_id, agv in self.world.get_component(Crane):

                    if info.offset not in self.dict_slot_cranes.keys():

                        self.dict_slot_cranes[info.offset] = []

                    if agv.min_offset <= info.offset <= agv.max_offset:

                        self.dict_slot_cranes[info.offset].append(agv_id)

    def get_start_slot(self) -> Slot:
        return self.world.component_for_entity(self.start_id, Slot)

    def get_slot(self, s_id) -> Slot:

        assert s_id > 0

        return self.world.component_for_entity(s_id, Slot)

    def get_slot_by_code(self, code) -> Tuple[int, Slot]:

        for s_id, s in self.world.get_component(Slot):

            if s.code == code:

                return s_id, self.world.component_for_entity(s_id, Slot)

        return 0, None

    def get_jobid_at_start(self) -> int:

        if wj := self.world.try_component(self.start_id, WithJob):

            return wj.job_id

        return 0

    def get_job_at_start(self) -> Job:

        return self.get_job(self.start_id)

    def get_job(self, s_id) -> Job:

        job = None

        if wj := self.world.try_component(s_id, WithJob):

            job = self.world.component_for_entity(wj.job_id, Job)

        return job

    def get_urgents(self) -> Tuple[int, Slot]:

        slots = []

        for s_id, (s, wait) in self.world.get_components(Slot, Wait):

            if self.world.has_component(s_id, Locked):
                continue

            if wait.left < NOTIFY_BEFORE_FINISH:

                slots.append((wait.left, s_id, s))

        return slots

    def job_next_pos(self, group: int, slot_id: int) -> Tuple[int]:

        slot = self.world.component_for_entity(slot_id, Slot)

        if wj := self.world.try_component(slot_id, WithJob):

            job = self.world.component_for_entity(wj.job_id, Job)

            info = self.job_mgr.get_job_op_info(job, True)

            for _, (s, _) in self.world.get_components(Slot, Idle):

                if info.code == s.op_code and group == s.group:

                    return (slot.offset, s.offset)

        return (slot.offset, 9999)

    def get_job_slot(self, j_id, s_id) -> Tuple[Job, Slot]:

        assert j_id > 0

        job = self.job_mgr.get_job(j_id)

        assert not job is None

        slot = self.get_slot(s_id)

        assert not slot is None

        return (job, slot)

    def add_job(self, j_id, s_id) -> bool:

        job, slot = self.get_job_slot(j_id, s_id)
        job.offset = slot.offset

        op_info = self.job_mgr.get_job_op_info(job, False)

        if slot.op_code != op_info.code:

            esper.dispatch_event(EVENT_GAME_OVER, '{job} MISMATCH {s}', -5)

            return False

        # start=self.get_slot(self.start_id)

        if 'S' == slot.op_code:

            job.start_time = self.nb_steps

            esper.dispatch_event(EVENT_OP_START, job, slot)

        if 'E' == slot.op_code:  # 全部处理完

            job.end_time = self.nb_steps

            esper.dispatch_event(EVENT_JOB_FINISHED, j_id, job)

            self.world.add_component(j_id, JobDone())

        else:

            # if 'T'==slot.op_code:

            #     print(slot)

            self.world.add_component(s_id, Wait(op_info.op_time))

            self.world.add_component(s_id, WithJob(j_id))

            if self.world.has_component(s_id, Idle):

                self.world.remove_component(s_id, Idle)

        return True

    def remove_job(self, j_id, s_id):

        if j_id < 1:

            return

        job, slot = self.get_job_slot(j_id, s_id)

        if w := self.world.try_component(s_id, Wait):

            op_info = self.job_mgr.get_job_op_info(job, False)

            if len(slot.op_code) > 1:

                if abs(w.left) < op_info.op_time*0.05:

                    esper.dispatch_event(EVENT_OP_FINISHED, job, slot)

                else:

                    esper.dispatch_event(EVENT_OP_STOPED, job, slot)

            job.op_index += 1
            self.world.remove_component(s_id, WithJob)
            self.world.remove_component(s_id, Wait)

        self.world.add_component(s_id, Idle())

        if self.world.has_component(s_id, Locked):

            self.world.remove_component(s_id, Locked)

    # def get_states(self,crane:Crane)->List[machine_info]:

    #     rt=[]

    #     for s_id, s in self.world.get_component(Slot):

    #         if s.group!=crane.group:continue

    #         if crane.min_offset<=s.offset<=crane.max_offset:

    #             rt.append(self.get_state(s_id,s))

    #     return rt

    # def get_states(self)->List[job_state]:

    #     rt=[]

    #     for s_id, s in self.world.get_component(Slot):

    #         rt.append(self.get_state(s_id,s))

    #     return rt

    # def get_state(self,slot_id:int,slot:Slot)->job_state:
    #     rt=job_state(*([0]*7))

    #     rt=rt._replace(type=2)
    #     rt=rt._replace(offset=slot.offset)

    #     if wj:=self.world.try_component(slot_id,WithJob):

    #         job=self.world.component_for_entity(wj.job_id,Job)

    #         info=self.job_mgr.get_op_info(job.proc_code,job.op_index)

    #         rt=rt._replace(job_num=int(job.code[1:]))

    #         rt=rt._replace(proc_num=int(job.proc_code[1:]))

    #         rt=rt._replace(op_num=job.op_index+1)

    #         wait=self.world.component_for_entity(slot_id,Wait)

    #         rt=rt._replace(plan_op_time=wait.duration)

    #         rt=rt._replace(op_time=wait.timer)

    #     return rt
