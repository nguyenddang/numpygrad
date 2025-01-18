from numpygrad import npg
from numpygrad.engine import Tensor
import torch
import numpy as np


torch.manual_seed(42)
np.random.seed(42)
def test_softmax():
    ta = torch.randn(3, requires_grad=True)
    tc = torch.nn.functional.softmax(ta, dim=-1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.softmax(a, dim=-1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(3, 4, requires_grad=True)
    tc = torch.nn.functional.softmax(ta, dim=-1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.softmax(a, dim=-1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(3, 4, 5, requires_grad=True)
    tc = torch.nn.functional.softmax(ta, dim=-1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.softmax(a, dim=-1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_mean():
    ta = torch.randn(3, 4, requires_grad=True)
    tc = ta.mean(dim=1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.mean(a, dim=1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10,3,7, requires_grad=True)
    tc = ta.mean(dim=0)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.mean(a, dim=0)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_sigmoid():
    ta = torch.randn(3, 4, requires_grad=True)
    tc = torch.sigmoid(ta)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.sigmoid(a)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_relu():
    ta = torch.randn(3, 4, requires_grad=True)
    tc = torch.nn.functional.relu(ta)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.relu(a)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_tanh():
    ta = torch.randn(3, 4, requires_grad=True)
    tc = torch.tanh(ta)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.tanh(a)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_gelu():
    ta = torch.randn(3, 4, requires_grad=True)
    tc = 0.5 * ta * (1 + torch.tanh(np.sqrt(2 / np.pi) * (ta + 0.044715 * ta**3)))
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = npg.gelu(a)
    c.backward()
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
def test_cross_entropy():
    ta = torch.randn(3, 4, requires_grad=True)
    tb = torch.randint(0, 4, (3,), dtype=torch.long)
    tc = torch.nn.functional.cross_entropy(ta, tb)
    tc.backward()
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.int64), requires_grad=True)
    c = npg.cross_entropy(a, b)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(32, 64, requires_grad=True)
    tb = torch.randint(0, 64, (32, ), dtype=torch.long)
    tc = torch.nn.functional.cross_entropy(ta, tb)
    tc.backward()
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.int64), requires_grad=True)
    c = npg.cross_entropy(a, b)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc.data, rtol=1e-5, atol=1e-6)
    
    
    

    
    
    