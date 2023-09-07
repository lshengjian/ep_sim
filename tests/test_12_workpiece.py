import  sys
from os import path

dir=path.abspath(path.dirname(__file__) + './..')
sys.path.append(dir)
from epsim.core import Workpiece,WorldObj
from epsim.core.componets import OpLimitData
from epsim.utils.onehost import type2onehot,op_ket2onehots
def test_onehot():
   assert list(type2onehot(0,3))==[0.0,0.0,0.0]
   assert list(type2onehot(1,3))==[1.0,0.0,0.0]
   assert list(type2onehot(2,3))==[0.0,1.0,0.0]
   assert list(type2onehot(3,3))==[0.0,0.0,1.0]
   assert list(op_ket2onehots(0,3,5))   == [0. ,0. ,0.   ,0. ,0. ,0. ,0. ,0.]
   assert list(op_ket2onehots(102,3,5)) == [1. ,0. ,0.   ,0. ,1. ,0. ,0. ,0.]
   assert list(op_ket2onehots(205,3,5)) == [0. ,1. ,0.   ,0. ,0. ,0. ,0. ,1.]

def test_auto_inc_id():
    wp=Workpiece.make_new('A')
    assert wp.id==1
    assert str(wp)=='A-1 (0,1)'
    wp=Workpiece.make_new('B')
    assert wp.id==1
    wp=Workpiece.make_new('B')
    assert wp.id==2
    assert str(wp)=='B-2 (0,1)'
    
def test_imgae():
    wp=Workpiece.make_new('A')
    img=wp.image
    h,w,c=img.shape
    assert c==3
    assert h==WorldObj.TILE_SIZE 
    assert w==WorldObj.TILE_SIZE    

def test_state():
    wp=Workpiece.make_new('A')
    op_start:OpLimitData=OpLimitData(0,'A',101)
    wp.target_op_limit=op_start
    print(wp.state2data())


if __name__ == "__main__":
    test_state()