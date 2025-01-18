import torch
import numpy as np
from numpygrad.engine import Tensor

torch.manual_seed(42)
np.random.seed(42)
def test_add():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta + tb + ta
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a + b + a
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 1.23
    tc = tb + ta + tb + tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = b + a + b + b
    c.backward()
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(2, 3, 3, requires_grad=True)
    tb = torch.randn(3, requires_grad=True)
    tc = tb + ta + tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = b + a + b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    

def test_mul_single():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 1.23
    tc = ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 2.765
    tc = tb * ta 
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = b * a
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    

def test_mul_pointwise():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(3, requires_grad=True)
    tc = tb * ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = b * a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    
def test_matmul():
    ta = torch.randn(10, 3, 4, requires_grad=True)
    tb = torch.randn(4, 10, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

    
    ta = torch.randn(10, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(1, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
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
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = a ** b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_exp():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.exp()
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.exp()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_sum():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sum()
    tc.backward()
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.sum()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sum(dim=1)
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.sum(dim=1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_log():
    ta = torch.rand(10, 3, 3, requires_grad=True)
    tc = ta.log()
    tc.backward(torch.ones_like(tc))
    
    a = Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.log()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    
def test_mini_mlp():
    w0 = torch.randn(3, 4, requires_grad=True)
    b0 = torch.randn(4, requires_grad=True)
    w1 = torch.randn(4, 5, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)
    
    x = torch.randn(10, 3)
    h = x @ w0 + b0
    logits = h @ w1 + b1
    logits.backward(torch.ones_like(logits))
    
    wt0 = Tensor(w0.detach().numpy().astype(np.float32), requires_grad=True)
    bt0 = Tensor(b0.detach().numpy().astype(np.float32), requires_grad=True)
    wt1 = Tensor(w1.detach().numpy().astype(np.float32), requires_grad=True)
    bt1 = Tensor(b1.detach().numpy().astype(np.float32), requires_grad=True)
    xt = Tensor(x.detach().numpy().astype(np.float32))
    
    ht = xt @ wt0 + bt0
    ht = ht
    logitst = ht @ wt1 + bt1
    logitst.backward()
    
    assert torch.allclose(torch.from_numpy(wt0.grad), w0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt0.grad), b0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(wt1.grad), w1.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt1.grad), b1.grad, rtol=1e-5, atol=1e-6)
    

