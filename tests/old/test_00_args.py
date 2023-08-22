import pytest
import random

parametrize_gamma = pytest.mark.parametrize("gamma",[.1, 0.5, 1.0])
parametrize_n = pytest.mark.parametrize("n", [1, 10, 20])

@pytest.fixture
def eps():
    return 10e-3

@parametrize_gamma
@parametrize_n
def test_search(eps, gamma, n):
    random.seed(eps)
    data=gamma*n
    print(gamma,n,data) # pytest -s
    assert 0.1<=data<=20  

    