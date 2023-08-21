import esper
import numpy as np
from esper import World
from epsim.core import *
from epsim.core.systems import *
from typing import List

class Simulator:
    def __init__(self, cfg: "DictConfig"):
        self.PROCS = cfg['PROCS']
        self.JOBS = cfg['JOBS']
        self.CRANES = cfg['CRANES']
        self.SLOTS = cfg['SLOTS']

        self.is_fail = False
        self.is_done = False
        self.state = None
        self.mask = None
        self.reward = 0.0
        self.total_reward = 0.0
        self.need_reset_sys = []
        world = World()
        self.job_mgr = JobMgr(world, cfg)
        self.slot_mgr = SlotMgr(world, self.job_mgr, cfg)
        self.crane_mgr = CraneMgr(world, self.slot_mgr, cfg)
        mgrs = (self.job_mgr, self.slot_mgr, self.crane_mgr)

        sys = SysTranslate(mgrs)
        self.need_reset_sys.append(sys)
        world.add_processor(sys, priority=10)
        world.add_processor(SysWorking(mgrs), priority=1)
        world.add_processor(SysMoveTo(mgrs))
        world.add_processor(SysUp(mgrs))
        world.add_processor(SysDown(mgrs))
        world.add_processor(SysAutoPick(mgrs))
        world.add_processor(SysAutoDrop(mgrs))
        world.add_processor(SysCount(mgrs))
        self.world = world
        esper.set_handler(EVENT_GAME_OVER, self.game_over)
        esper.set_handler(EVENT_OP_FINISHED, self.op_finished)
        esper.set_handler(EVENT_OP_STOPED, self.op_stoped)
        esper.set_handler(EVENT_JOB_FINISHED, self.job_finished)

    def next_proc_key(self):
        return self.job_mgr.get_proc_index()

    def reset(self, seed=None):
        debug('******reset!*******')
        if seed:
            self.seed(seed)

        self.is_fail = False
        self.is_done = False
        self.reward = 0.0
        self.total_reward = 0.0
        self.world.clear_cache()
        self.world.clear_database()
        self.job_mgr.make_jobs(self.JOBS)
        self.crane_mgr.make_cranes()
        self.slot_mgr.make_slots()
        for sys in self.need_reset_sys:
            sys.reset()

    def op_stoped(self, job, slot):
        debug(f'{slot} stop {job}!')
        self.reward = -5.0

        # self.total_reward+=self.reward

    def op_finished(self, job, slot):
        debug(f'{slot} finished op!')
        self.reward = 1.0
        # self.total_reward+=self.reward

    def game_over(self, msg, r=-1):
        debug(f'game over!--{msg}')
        # debug(self.mask)
        self.reward = r
        # self.total_reward+=self.reward
        self.is_fail = True

    def can_add_job(self):
        l1 = len(self.slot_mgr.get_slot_jobs(True))
        l2 = len(self.crane_mgr.get_crane_jobs(True))
        # print(l1,l2,self.crane_mgr.nb_cranes_first_group)
        return l1 + l2 < self.crane_mgr.nb_cranes_first_group

    def job_finished(self, _, job):
        debug(f'finished:{job}!')
        self.reward = 10.0
        total = self.job_mgr.get_num_todo_jobs() + \
            len(self.slot_mgr.get_slot_jobs()) + \
            len(self.crane_mgr.get_crane_jobs())

        if total < 1:
            self.total_reward += len(self.job_mgr.ids)*5
            print(f'===all jobs finished===,total_reward:{self.total_reward}')
            self.is_done = True

    def step(self, action, agent=None):
        if self.is_done or self.is_fail:
            return
        self.reward = 0
        if agent is None or agent == 'dispatch':
            if action == 1:
                self.job_mgr.get_proc_index(True)

            elif action == 2:
                job = self.slot_mgr.get_job_at_start()
                if job:
                    self.game_over('start slot had job!', -10)
                    self.total_reward += self.reward
                    return

                j_id, job = self.job_mgr.take_away(self.job_mgr.cur_proc_key)
                if j_id > 0:
                    self.slot_mgr.add_job(j_id, self.slot_mgr.start_id)

            else:
                self.reward -= 0.01

        else:
            if action != 0:
                # print(agent,action)
                self.do_cmd(agent, action)
        self.slot_mgr.step()
        self.total_reward += self.reward
        self.mask = self.get_mask(self.state)

    def do_cmd(self, agent, step):
        for agv_id, agv in self.world.get_component(Crane):
            if agv.code == agent:
                self.world.add_component(agv_id, MoveTo(agv.offset+step))
                if self.world.has_component(agv_id, Idle):
                    self.world.remove_component(agv_id, Idle)
                    break

    def seed(self, seed=None):
        pass

    def get_mask(self, state=None):
        rt = np.zeros(3, dtype=np.float32)
        rt[0] = 1
        f1 = self.can_add_job()
        f2 = self.slot_mgr.get_job_at_start() is None
        f3 = len(self.job_mgr.todo_ids) > 0

        if f3:
            rt[1] = 1

        flag = f1 and f2 and f3
        if flag:
            rt[2] = 1

        self.mask = rt
        # print(rt)
        return rt

    def get_state(self) -> List[job_state]:
        rt=self.job_mgr.get_states()
        #rt.extend(self.crane_mgr.get_states())
        #rt.extend(self.slot_mgr.get_states())
        #rt = sorted(rt, key=lambda x: x.offset)
        cnt = len(rt)
        if cnt < MAX_JOBS:
            # print(f'{cnt}<{VIEW_DISTANCE}')
            for i in range(MAX_JOBS-cnt):
                rt.append([0]*6)

        return rt
