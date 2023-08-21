import pytest
import numpy as np
import torch


def test_demo():
    assert 1+1 == 2
    assert round(1.4)==1
    assert round(1.8)==2
    assert round(3.5)==4
    assert round(4.5)==4

def test_numpy_statck():
    data=np.array([1,2,3,4])
    data=np.stack([data]*2,axis=0)
    assert data.shape == (2,4)
    data=np.stack([data]*3,axis=0)
    assert data.shape == (3,2,4)

def test_onehot():
    indexs = [1, 0, 3]
    n_categaries = 5#np.max(indexs) + 1
    print(np.eye(n_categaries)[indexs])
    
def test_tensor():
    v1=torch.tensor([[1],[2]]) #矢量按照列排
    f1=v1.T @ v1
    assert f1.shape==(1,1) and f1.item()==5
    m1=v1 @ v1.T
    assert m1.shape==(2,2) and (m1==torch.tensor([[1,2],[2,4]])).all()

    tensor=torch.tensor([[1,2],[3,4]])
    tensor1=tensor.T
    assert (tensor1==torch.tensor([[1,3],[2,4]])).all()
    y1 = tensor @ tensor.T
    assert (y1==torch.tensor([[5,11],[11,25]])).all()
    y2 = tensor.matmul(tensor.T)
    assert (y1==y2).all()

    y3 = torch.zeros_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)
    assert (y1==y3).all()


    # # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    assert (z1==z2).all()

    z3 = torch.zeros_like(tensor)
    torch.mul(tensor, tensor, out=z3)