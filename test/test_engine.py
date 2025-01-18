import torch
import numpy as np
from numpygrad.engine import Tensor

# Test individual fn first
torch.manual_seed(42)
np.random.seed(42)
def test_add():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta + tb + ta
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = Tensor(tb.detach().numpy().astype(np.float32))
    c = a + b + a
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

def test_mul_single():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 1.23
    tc = ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = tb
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

def test_mul_pointwise():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = Tensor(tb.detach().numpy().astype(np.float32))
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 2.765
    tc = ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = tb
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    
def test_matmul():
    ta = torch.randn(10, 3, 4, requires_grad=True)
    tb = torch.randn(4, 10, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = Tensor(tb.detach().numpy().astype(np.float32))
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

    
    ta = torch.randn(10, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = Tensor(tb.detach().numpy().astype(np.float32))
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(1, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = Tensor(tb.detach().numpy().astype(np.float32))
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

def test_pow():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 5
    tc = ta ** tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    b = tb
    c = a ** b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_sigmoid():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sigmoid()
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    c = a.sigmoid()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_tanh():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.tanh()
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    c = a.tanh()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_relu():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.relu()
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32))
    c = a.relu()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_mini_mlp():
    w0 = torch.randn(3, 4, requires_grad=True)
    b0 = torch.randn(4, requires_grad=True)
    w1 = torch.randn(4, 5, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)
    
    x = torch.randn(10, 3)
    y = torch.randn(10, 5)
    h = torch.relu(x @ w0 + b0)
    logits = h @ w1 + b1
    loss = ((logits - y) ** 2).mean()
    loss.backward()
    
    wt0 = Tensor(w0.detach().numpy().astype(np.float32))
    bt0 = Tensor(b0.detach().numpy().astype(np.float32))
    wt1 = Tensor(w1.detach().numpy().astype(np.float32))
    bt1 = Tensor(b1.detach().numpy().astype(np.float32))
    xt = Tensor(x.detach().numpy().astype(np.float32))
    yt = Tensor(y.detach().numpy().astype(np.float32))
    
    ht = xt @ wt0 + bt0
    ht = ht.relu()
    logitst = ht @ wt1 + bt1
    loss = ((logitst - yt) ** 2).sum() / (10 * 5)
    
    assert torch.allclose(torch.from_numpy(wt0.grad), w0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt0.grad), b0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(wt1.grad), w1.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt1.grad), b1.grad, rtol=1e-5, atol=1e-6)
    

