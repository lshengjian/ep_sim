from ..core.componets import *
from ..base_sys import SysBase
from ..utils import *
from ..utils.consts  import MAX_WORK_JOBS,MIN_WAIT_TIME
import esper

class SysPickJob(SysBase):
    def __init__(self,game):
        super().__init__(game)
        self.start_id=game.start_id
        self.timer=0
        esper.set_handler('game_over', self.reset)



    def reset(self,msg,r):
        self.timer=0
    def process(self):
        self.timer=(self.timer+1)%(MIN_WAIT_TIME*2)
        if self.timer!=1:
            return
        
        todos=self.game.get_todo_jobs()
        if len(todos)<1:
            #print('no todo jobs!')
            return
        nb_jobs=min(MAX_WORK_JOBS,self.game.nb_start_cranes)
        if len(self.game.get_slot_jobs(True))+len(self.game.get_crane_jobs(True))>=nb_jobs:
            #print('too much jobs!')
            return #正在作业的物料太多了

        times=[]#找最剩余时间最短的加工槽进行处理
        for _, (s,wait,_) in self.world.get_components(Slot,Wait,WithJob):
            times.append(wait.left)
        times.sort()
        
        job_id,job=todos[0]
        if self.world.has_component(self.start_id,Idle):
            todos.pop(0)
            self.world.remove_component(job_id,Idle)
            #print(f'put {job} on start')
            self.world.remove_component(self.start_id,Idle)
            info=self.get_op_info(job.proc_code,0)
            self.world.add_component(self.start_id,Wait(info.op_time))
            self.world.add_component(self.start_id,WithJob(job_id))
            start=self.world.component_for_entity(self.start_id,Slot)
            job.offset=start.offset

        




        





                


