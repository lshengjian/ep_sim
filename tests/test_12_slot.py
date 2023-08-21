import  sys
from os import path
from esper import  World
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core.componets import *
from epsim.core import Simulator
from epsim.utils import load_config

def test_start_job():
    args=load_config('demo-1a.yaml')
    game=Simulator(args,False)
    game.reset()
    jm=game.job_mgr
    sm=game.slot_mgr
    world=game.world
    slot=sm.get_slot(sm.start_id)
    assert  world.has_component(sm.start_id,Idle)
    assert slot.op_code=='S'
    j_id,job=jm.take_away('P1')
    sm.step()
    assert sm.nb_steps==1
    assert sm.add_job(j_id,sm.start_id)
    assert job.start_time==1

    assert not world.has_component(j_id,Idle)
    assert not world.has_component(sm.start_id,Idle)
    assert  world.has_component(sm.start_id,Wait)
    assert  world.has_component(sm.start_id,WithJob)
    wj=world.component_for_entity(sm.start_id,WithJob)
    assert wj.job_id==j_id
    wt=world.component_for_entity(sm.start_id,Wait)
    assert wt.timer==0
    sm.step()
    assert wt.timer==1
    sm.remove_job(j_id,sm.start_id)
    info=jm.get_job_op_info(job,False)
    assert job.op_index==1
    assert info.code=='SX'

def loop(sm,time):
    for t in range(time):
        sm.step()
def test_finish_job():
    args=load_config('demo-1a.yaml')
    game=Simulator(args,False)
    game.reset()
    jm=game.job_mgr
    sm=game.slot_mgr
    world=game.world
    start=sm.get_slot(sm.start_id)
    j_id,job=jm.take_away('P1')
    sm.step()
    sm.add_job(j_id,sm.start_id)
    sm.step()
    info=jm.get_job_op_info(job,False)
    assert job.op_index==0
    assert info.code=='S'
    sm.remove_job(j_id,sm.start_id)
    info=jm.get_job_op_info(job,False)
    assert job.op_index==1
    assert info.code=='SX'
    s_id,slot=sm.get_slot_by_code('SX1')
    sm.add_job(j_id,s_id)
    loop(sm,40)
    sm.remove_job(j_id,s_id)
    assert job.op_index==2
    info=jm.get_job_op_info(job,False)
    assert info.code=='E'
    s_id,slot=sm.get_slot_by_code('E')
    sm.add_job(j_id,s_id)
    loop(sm,1)
    assert job.op_index==2
    assert job.end_time>40
    assert world.has_component(j_id,JobDone)




    

