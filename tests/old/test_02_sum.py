import numpy as np
import torch

def test_einsum():
    # batch permute
    A = torch.randn(2, 3, 4, 5)
    print(torch.einsum('...ij->...ji', A).shape)
    
def test_diagonal():
    
    # trace
    a=torch.randn(4, 4)
    print(a.numpy())
    d=torch.einsum('ii',a)
    print(d.numpy())
    assert d.shape == torch.Size([]) 

    # diagonal
    a=torch.randn(4, 4)
    print(a.numpy())
    d=torch.einsum('ii->i',a)
    print(d.numpy())
    assert d.shape == (4,)

def test_product():
    # outer product
    x = torch.randn(5)
    print(x.numpy())
    y = torch.randn(2)
    print(y.numpy())
    d=torch.einsum('i,j->ij', x, y)
    print(d.numpy())
    assert d.shape == (5,2)
    # batch matrix multiplication
    As = torch.randn(3,2,3)
    print(As.numpy())
    Bs = torch.randn(3,3,4)
    print(Bs.numpy())
    Cs=torch.einsum('bij,bjk->bik', As, Bs)
    print(Cs.numpy())
    assert Cs.shape == (3,2,4)