import  sys
from os import path
from esper import  World
dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core.componets import *
from epsim.core.jobs import JobMgr
from epsim.utils import load_config
def test_take_away():
    world=World()
    args=load_config('demo-2a.yaml')
    jm=JobMgr(world,args)
    jm.make_jobs({ 'P1': 1,'P2': 2 })
    stat=jm.stat
    assert stat=={ 'P1': 1,'P2': 2 }

    _,job=jm.take_away('P1')
    assert job.proc_code == 'P1'
    #stat=jm.job_stat
    assert stat=={ 'P1': 0,'P2': 2 }

    jm.take_away('P2')
    _,job=jm.take_away('P2')
    assert job.proc_code == 'P2'
    jid,job=jm.take_away('P2')
    assert jid == 0
    assert job is None
    #stat=jm.job_stat
    assert stat=={ 'P1': 0,'P2': 0 }

def test_ok_proc():
    world=World()
    args=load_config('demo-2a.yaml')
    jm=JobMgr(world,args)
    jm.make_jobs({ 'P1': 1,'P2': 2 })
    assert jm.cur_proc_key=='P1'
    jm.get_proc_index(True)
    assert jm.cur_proc_key=='P2'
    jm.get_proc_index(True)
    assert jm.cur_proc_key=='P1'
    p_code=jm.procs[0]
    jm.take_away(p_code)
    assert jm.stat=={ 'P1': 0,'P2': 2 }
    idx=jm.get_proc_index(False)
    assert idx==1
    idx=jm.get_proc_index(True)
    assert idx==1
    p_code=jm.procs[idx]
    jm.take_away(p_code)
    assert jm.stat=={ 'P1': 0,'P2': 1 }



    

