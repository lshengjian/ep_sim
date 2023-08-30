import numpy as np
def test_nozero_indexs():
    data=np.array([1.0,0,3.0,-2.0,0.0])
    idxs=np.argwhere(data != 0)
    assert list(idxs)==[0,2,3]