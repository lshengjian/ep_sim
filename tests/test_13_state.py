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
    args=load_config('demo-1a.yaml')  #P1:2
    game=Simulator(args,False)
    game.reset()
    data=game.get_state()
    assert len(data)==MAX_JOBS
    d=data[0]
    print(d)
    assert d.offset==0
    assert d.job_num==1
    assert d.proc_num==1
    assert d.op_num==1
    assert d.plan_op_time>0
    assert d.op_time<1
    game.step(2)
    sm=game.slot_mgr
    loop(sm,1)
    data=game.get_state()
    d=data[0]
    print(d)
    assert d.offset==1
    assert d.op_time>0

    

    
    
