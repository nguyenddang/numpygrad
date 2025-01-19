import torch
import numpy as np
import npg

'''Test npg.Tensor basic operations'''
torch.manual_seed(42)
np.random.seed(42)
def test_add():
    ta = torch.randn(2, 3, 3, requires_grad=True)
    tb = torch.randn(2, 3, 3, requires_grad=True)
    tc = ta + tb + ta
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a + b + a
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 1.23
    tc = tb + ta + tb + tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = b + a + b + b
    c.backward()
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(2, 3, 3, requires_grad=True)
    tb = torch.randn(3, requires_grad=True)
    tc = tb + ta + tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
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
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = 2.765
    tc = tb * ta 
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
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
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a * b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(3, requires_grad=True)
    tc = tb * ta * tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
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
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)

    
    ta = torch.randn(10, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(1, 3, 4, 6, requires_grad=True)
    tb = torch.randn(10, 3, 6, 4, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
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
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = tb
    c = a ** b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_exp():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.exp()
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.exp()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_log():
    ta = torch.rand(10, 3, 3, requires_grad=True)
    tc = ta.log()
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.log()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_sum():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sum()
    tc.backward()
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.sum()
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sum(dim=1)
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.sum(dim=1)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tc = ta.sum(dim=1, keepdim=True)
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.sum(dim=1, keepdim=True)
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_reshape():
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb  = torch.randn(30, 3, requires_grad=True)
    tc = ta.reshape(30, 3) + tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.reshape(30, 3) + b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(90, requires_grad=True)
    tc = ta.reshape(-1) + tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.reshape(-1) + b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_split():
    qkv = torch.randn(384, 384, requires_grad=True)
    x = torch.randn(2, 384)
    attn = x @ qkv
    q, k, v = attn.split(128, dim=-1)    
    sum = q + k + v
    sum.backward(torch.ones_like(sum))
    
    qkvn = npg.Tensor(qkv.detach().numpy().astype(np.float32), requires_grad=True)
    xn = npg.Tensor(x.detach().numpy().astype(np.float32))
    attnn = xn @ qkvn
    qn, kn, vn = attnn.split(128, dim=-1)
    sumn = qn + kn + vn
    sumn.backward()
    assert torch.allclose(torch.from_numpy(qkvn.grad), qkv.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(qn.data), q, rtol=1e-5, atol=1e-4)
    assert torch.allclose(torch.from_numpy(kn.data), k, rtol=1e-5, atol=1e-4)
    assert torch.allclose(torch.from_numpy(vn.data), v, rtol=1e-5, atol=1e-4)
    assert torch.allclose(torch.from_numpy(sumn.data), sum, rtol=1e-5, atol=1e-4)
    
def test_transpose():
    
    ta = torch.randn(10, 3, 3, requires_grad=True)
    tb = torch.randn(3, 10, requires_grad=True)
    tat = ta.transpose(0, 1)
    tc = tat @ tb
    tc.backward(torch.ones_like(tc))
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    at = a.transpose(0, 1)
    c = at @ b
    c.backward()
    
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(c.data), tc, rtol=1e-5, atol=1e-6)
    
def test_masked_fill():
    ta = torch.randint(0, 2, (2, 3, 3)).float()
    ta.requires_grad = True
    tb = torch.randn(2, 3, 3, requires_grad=True) 
    tc = ta.masked_fill(ta== 0.0, 3.432) 
    td = tc + tb
    sum = torch.sum(td)
    sum.backward()
    
    a = npg.Tensor(ta.detach().numpy().astype(np.float32), requires_grad=True)
    b = npg.Tensor(tb.detach().numpy().astype(np.float32), requires_grad=True)
    c = a.masked_fill(a==0.0, 3.432)
    d = c + b
    sumn = npg.sum(d)
    sumn.backward()
    assert torch.allclose(torch.from_numpy(a.grad), ta.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(b.grad), tb.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(d.data), td, rtol=1e-5, atol=1e-4)
    
def test_mini_mlp():
    w0 = torch.randn(3, 4, requires_grad=True)
    b0 = torch.randn(4, requires_grad=True)
    w1 = torch.randn(4, 5, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)
    
    x = torch.randn(10, 3)
    h = x @ w0 + b0
    logits = h @ w1 + b1
    logits.backward(torch.ones_like(logits))
    
    wt0 = npg.Tensor(w0.detach().numpy().astype(np.float32), requires_grad=True)
    bt0 = npg.Tensor(b0.detach().numpy().astype(np.float32), requires_grad=True)
    wt1 = npg.Tensor(w1.detach().numpy().astype(np.float32), requires_grad=True)
    bt1 = npg.Tensor(b1.detach().numpy().astype(np.float32), requires_grad=True)
    xt = npg.Tensor(x.detach().numpy().astype(np.float32))
    
    ht = xt @ wt0 + bt0
    ht = ht
    logitst = ht @ wt1 + bt1
    logitst.backward()
    
    assert torch.allclose(torch.from_numpy(wt0.grad), w0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt0.grad), b0.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(wt1.grad), w1.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(bt1.grad), b1.grad, rtol=1e-5, atol=1e-6)
    
