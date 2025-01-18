import npg
import torch
import numpy as np

def test_MLP_w_Layernorm():
    xn = npg.randn(10, 3)
    yn = npg.randn(10, 5)
    xt = torch.from_numpy(xn.data)
    yt = torch.from_numpy(yn.data)
    l1n = npg.nn.Linear(3, 5)
    l2n = npg.nn.Linear(5, 5)
    l1t = torch.nn.Linear(3, 5)
    l2t = torch.nn.Linear(5, 5)
    lnn = npg.nn.Layernorm(3)
    ltn = torch.nn.LayerNorm(3)
    ltn.weight.data = torch.from_numpy(lnn.weight.data)
    ltn.bias.data = torch.from_numpy(lnn.bias.data)
    l1t.weight.data = torch.from_numpy(l1n.W.data.T)
    l1t.bias.data = torch.from_numpy(l1n.b.data)
    l2t.weight.data = torch.from_numpy(l2n.W.data.T)
    l2t.bias.data = torch.from_numpy(l2n.b.data)
    
    ln1_out = lnn(xn)
    l1n_out = l1n(ln1_out)
    relu_out_n = npg.relu(l1n_out)
    logitsn = l2n(relu_out_n)
    lossn = npg.mse_loss(logitsn, yn)
    
    ln1_out_t = ltn(xt)
    l1t_out = l1t(ln1_out_t)
    relu_out_t = torch.relu(l1t_out)
    logitst = l2t(relu_out_t)
    losst = torch.nn.functional.mse_loss(logitst, yt)
    
    lossn.backward()
    losst.backward()
    assert torch.allclose(torch.from_numpy(l1n.W.grad), l1t.weight.grad.T, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(l1n.b.grad), l1t.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(l2n.W.grad), l2t.weight.grad.T, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(l2n.b.grad), l2t.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(l1n.W.data), l1t.weight.data.T, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(l1n.b.data), l1t.bias.data, rtol=1e-5, atol=1e-6)
    

    
    
    