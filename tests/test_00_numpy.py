import numpy as np

def test_slice():
    data=np.array([1.,2.,3.])
    d2=data[:2] # ref
    d2[:]*=2

    assert d2.tolist()==[2.,4.]
    assert data.tolist()==[2.,4.,3.]

    data=[1,2,3]
    d2=data[:2] # copy
    d2[0]=2.0
    d2[1]=4.0
    assert d2==[2.,4.]
    assert data==[1.,2.,3.]

def test_nozero_indexs():
    data=np.array([1.0,0,3.0,-2.0,0.0])
    idxs=np.argwhere(data != 0)
    assert list(idxs)==[0,2,3]