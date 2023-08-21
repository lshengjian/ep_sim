import esper
import numpy as np
from typing import  Tuple,List,Dict
from epsim.core import Job,Count,Idle,op_info,job_state,EVENT_OP_START,EVENT_OP_DOING
from copy import deepcopy
class JobMgr:
    def __init__(self,world,args):
        self.PROCS=args['PROCS']
        self.world =world
        self.todo_ids=[]
        self.ids=[]
        self.dict_proc_jobs=dict()
        self.cur_proc_key=None
        
        #esper.set_handler(EVENT_GAME_OVER, self.reset)
        esper.set_handler(EVENT_OP_START, self.op_start)
        esper.set_handler(EVENT_OP_DOING, self.op_doing)
    
    def get_proc_steps(self,proc_code:str):
        '''
        获取流程有几个处理步骤
        '''
        return len(self.PROCS[proc_code])

    def max_steps(self):
        rt=0
        for j_id in self.ids:
            job=self.get_job(j_id)
            step=self.get_proc_steps(job.proc_code)
            if rt<step: rt=step
        return rt



    def op_start(self,job,slot):
        #job=self.game.get_job(job_id)
        job_idx=int(job.code[1:])-1
        step=job.op_index
        self.data[job_idx,step,1]=0

    def op_doing(self,job,slot):
        job_idx=int(job.code[1:])-1
        step=job.op_index
        self.data[job_idx,step,1]+=1

    @property 
    def procs(self)->List[str]:
        return list(self.dict_proc_jobs.keys())

    @property 
    def stat(self)->Dict[str,int]:
        # rt=dict()
        # for j_id in  self.job_ids:
        #     job:Job=self.get_job(j_id)
        #     if job.proc_code not in rt:
        #         rt[job.proc_code]=0
        #     rt[job.proc_code]+=1
        return self.dict_proc_jobs

    def _proc_index(self,todo_procs:List[str],next=True)->int:
        if self.cur_proc_key not in todo_procs:
            self.cur_proc_key=None if len(todo_procs)<1 else todo_procs[0]
            return -1
        idx=todo_procs.index(self.cur_proc_key)
        if next:
            idx=(idx+1)%len(todo_procs)
        p_code=todo_procs[idx]
        for i,k in enumerate(self.stat.keys()):
            if k==p_code:
                self.cur_proc_key=p_code
                return i
        return -1

    def get_proc_index(self,next=True)->int:
        todos=[]
        for k,v in self.stat.items():
            if v>0:
                todos.append(k)
        if  len(todos)<1:
            self.cur_proc_key=None
            return -1
        if  self.cur_proc_key is None:
            self.cur_proc_key=todos[0]
            return self._proc_index(todos,False)
        return self._proc_index(todos,next)



    def make_jobs(self,jobs:Dict[str,int]):
        self.todo_ids=[]
        self.ids=[]
        self.dict_proc_jobs=deepcopy(jobs)
        js=[]
        
        for (k,v) in jobs.items():
            js+=[k]*v
            
        total=sum(jobs.values())
        assert total==len(js)
        
        for idx,proc_code in enumerate(js):
            job=Job(f'J{idx+1}',proc_code)
            j_id=self.world.create_entity(job,Count(),Idle())
            self.ids.append(j_id)
            self.todo_ids.append(j_id)
            
        self.cur_proc_key=js[0]
        self.data=np.zeros((len(self.ids),self.max_steps(),2),dtype=np.uint16) #计划，实际处理时间
        for j_id in self.ids:
            job=self.get_job(j_id)
            job_idx=int(job.code[1:])-1
            for t in range(self.get_proc_steps(job.proc_code)):
                info=self.PROCS[job.proc_code][t]
                #print(job_idx,t)
                self.data[job_idx,t,0]=info.op_time
        
        #self.data[:,:,1] =0
        


    def take_away(self,proc_code:str)->Tuple[int,Job]:
        if len(self.todo_ids)<1 or self.dict_proc_jobs[proc_code]<1:
            self.cur_proc_key=None
            return (0,None)
        stat=self.dict_proc_jobs
        stat[proc_code]-=1
        j_idx=-1
        for idx,j_id in enumerate(self.todo_ids):
            job=self.get_job(j_id)
            if job.proc_code==proc_code:
                j_idx=idx
                break
        
        if j_idx>=0:
            job_id=self.todo_ids.pop(j_idx)
            job=self.get_job(job_id)
            self.world.remove_component(job_id,Idle)
            self.get_proc_index(False)
            return (job_id,job)  
        self.cur_proc_key=None
        return (0,None)

    def get_job(self,job_id:int)->Job:
        return self.world.component_for_entity(job_id,Job)
    
    def get_num_todo_jobs(self)->int:
        return len(self.todo_ids)

    def get_job_op_info(self,job:Job,is_next=False)->op_info:
        PROCS=self.PROCS
        op_index=job.op_index
        proc_code=job.proc_code
        if is_next:op_index+=1
        if op_index>=len(PROCS[proc_code]):
            op_index=len(PROCS[proc_code])-1
        return PROCS[proc_code][op_index]
    

    def get_op_info(self,proc_code:str,op_index:int)->op_info:
        PROCS=self.PROCS
        if op_index>=len(PROCS[proc_code]):
            op_index=len(PROCS[proc_code])-1
        return PROCS[proc_code][op_index]  

    def get_states(self)->List[job_state]:
        rt=[]
        for job_id in self.ids:
            job=self.get_job(job_id)
            rt.append(self.get_state(job))
        return rt  

    def get_state(self,job:Job)->job_state:
        rt=job_state(*([0]*6))
        rt=rt._replace(job_num=int(job.code[1:]))
        rt=rt._replace(proc_num=int(job.proc_code[1:]))
        rt=rt._replace(op_num=job.op_index+1)
        ts=self.data[rt.job_num-1][job.op_index]
        rt=rt._replace(plan_op_time=ts[0])
        rt=rt._replace(op_time=ts[1])
        rt=rt._replace(offset=job.offset)

        return rt