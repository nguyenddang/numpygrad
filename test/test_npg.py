from numpygrad import npg
from numpygrad.engine import Tensor
import torch
import numpy as np


torch.manual_seed(42)
np.random.seed(42)
def test_softmax():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = torch.nn.functional.softmax(ta, dim=-1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    c = npg.softmax(a, dim=-1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    