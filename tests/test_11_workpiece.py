import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import Workpiece


def test_auto_inc_id():
    wp=Workpiece.make_new('A')
    assert wp.id==1
    assert str(wp)=='A-1 (0,1)'
    wp=Workpiece.make_new('B')
    assert wp.id==1
    wp=Workpiece.make_new('B')
    assert wp.id==2
    assert str(wp)=='B-2 (0,1)'
    
    

