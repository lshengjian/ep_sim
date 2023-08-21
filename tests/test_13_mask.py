import  sys,esper
from os import path
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import *
from epsim.utils import load_config

def loop(sm,time):
    for t in range(time):
        sm.step()
def test_ok():
    args=load_config('demo-1a.yaml')
    game=Simulator(args,False)
    game.reset()
    sm=game.slot_mgr
    mask=game.get_mask()
    assert sm.job_mgr.cur_proc_key =='P1'
    assert len(mask)==3
    assert all(mask)
    assert len(sm.job_mgr.todo_ids)==2

    game.step(2)
    loop(sm,3) 
    mask=game.mask
    assert all(mask[:2])
    assert mask[2]==0
    assert len(sm.job_mgr.todo_ids)==1

    job_id=sm.get_jobid_at_start()
    sm.remove_job(job_id,sm.start_id)
    loop(sm,3) 
    mask=game.get_mask()
    assert all(mask)

    game.step(2) 
    assert sm.job_mgr.cur_proc_key is None
    loop(sm,3) 
    mask=game.mask
    assert not any(mask[1:])
    assert len(sm.job_mgr.todo_ids)==0
    
    job_id=sm.get_jobid_at_start()
    sm.remove_job(job_id,sm.start_id)
    
    game.step(2) # is ok
 

def test_fail():
    args=load_config('demo-1a.yaml')
    game=Simulator(args,False)
    game.reset()
    sm=game.slot_mgr
    mask=game.get_mask()
    assert sm.job_mgr.cur_proc_key =='P1'
    assert len(mask)==3
    assert all(mask)
    assert len(sm.job_mgr.todo_ids)==2

    game.step(2)
    loop(sm,3) 
    mask=game.mask
    assert all(mask[:2])
    assert mask[2]==0
    assert len(sm.job_mgr.todo_ids)==1

    game.step(2) # left 0
    assert game.is_fail==True





    




    

