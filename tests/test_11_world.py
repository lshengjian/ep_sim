import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import build_config,World,Workpiece,CraneAction

def test_build_world():
    wd=World('test01',False)
    assert len(wd.group_cranes)==1
    x1,x2=wd.group_limits[1]
    assert x1==1 and x2==9
    # x1,x2=wd.group_limits[2]
    # assert x1==20 and x2==31
    # assert wd.group_slots[1][-1].x+1==wd.group_slots[2][0].x # swap


def test_set_products():
    wd=World('test01',True)
    wd.reset()
    wd.add_jobs(['A']*3)
    assert len(wd.products)==2
    assert wd.starts[0].carrying!=None
    assert wd.starts[1].carrying is None

    assert len(wd.pos_slots)==7
    assert len(wd.product_procs['A'])==3
    wd.pprint()

if __name__ == "__main__":
    test_set_products()    
    

