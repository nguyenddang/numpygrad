import npg
import torch
import numpy as np
from demo.nanogpt.model import CausualSelfAttention, MLP, Block
from model_torch import CausalSelfAttentionTorch, GPTConfig, LayerNormTorch, MLPTorch, BlockTorch

def test_attention():
    config = GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0)
    blockt = CausalSelfAttentionTorch(config)
    blockn = CausualSelfAttention(config)
    
    blockn.c_attn.weight = npg.Tensor(blockt.c_attn.weight.detach().numpy(), requires_grad=True)
    blockn.c_attn.bias = npg.Tensor(blockt.c_attn.bias.detach().numpy(), requires_grad=True)
    blockn.c_proj.weight = npg.Tensor(blockt.c_proj.weight.detach().numpy(), requires_grad=True)
    blockn.c_proj.bias = npg.Tensor(blockt.c_proj.bias.detach().numpy(), requires_grad=True)
    
    xt = torch.randn(2, 4, 4)
    yt = torch.randint(0, 4, (2, 4), dtype=torch.long)
    xn = npg.Tensor(xt.detach().numpy().astype(np.float32))
    yn = npg.Tensor(yt.detach().numpy().astype(np.int32))
    
    logitst = blockt(xt)
    losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), yt.view(-1))
    losst.backward()
    
    logitsn = blockn(xn)
    lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
    lossn.backward()
    
    assert torch.allclose(torch.from_numpy(blockn.c_attn.weight.grad), blockt.c_attn.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.c_attn.bias.grad), blockt.c_attn.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.c_proj.weight.grad), blockt.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.c_proj.bias.grad), blockt.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)

def test_layernorm():
    lnt = LayerNormTorch(4, bias=True)
    lnn = npg.nn.LayerNorm(4, bias=True)
    mlpt = MLPTorch(GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0))
    mlpn = MLP(GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0))
    # make sure the weights are the same
    lnn.weight = npg.Tensor(lnt.weight.detach().numpy(), requires_grad=True)
    lnn.bias = npg.Tensor(lnt.bias.detach().numpy(), requires_grad=True)
    mlpn.c_fc.weight = npg.Tensor(mlpt.c_fc.weight.detach().numpy(), requires_grad=True)
    mlpn.c_fc.bias = npg.Tensor(mlpt.c_fc.bias.detach().numpy(), requires_grad=True)
    mlpn.c_proj.weight = npg.Tensor(mlpt.c_proj.weight.detach().numpy(), requires_grad=True)
    mlpn.c_proj.bias = npg.Tensor(mlpt.c_proj.bias.detach().numpy(), requires_grad=True)
    
    xt = torch.randn(2, 4, 4)
    target = torch.randint(0, 4, (2, 4), dtype=torch.long)
    xn = npg.Tensor(xt.detach().numpy().astype(np.float32))
    yn = npg.Tensor(target.detach().numpy().astype(np.int32))
    
    logitst = mlpt(lnt(xt))
    losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), target.view(-1))
    losst.backward()
    
    logitsn = mlpn(lnn(xn))
    lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
    lossn.backward()
    
    assert torch.allclose(torch.from_numpy(lnn.weight.grad), lnt.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lnn.bias.grad), lnt.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    
def test_mlp():
    config = GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0)
    mlpn = MLP(config)
    mlpt = MLPTorch(config)
    
    mlpn.c_fc.weight = npg.Tensor(mlpt.c_fc.weight.detach().numpy(), requires_grad=True)
    mlpn.c_fc.bias = npg.Tensor(mlpt.c_fc.bias.detach().numpy(), requires_grad=True)
    mlpn.c_proj.weight = npg.Tensor(mlpt.c_proj.weight.detach().numpy(), requires_grad=True)
    mlpn.c_proj.bias = npg.Tensor(mlpt.c_proj.bias.detach().numpy(), requires_grad=True)
    
    x = torch.randn(2, 4, 4)
    target = torch.randint(0, 4, (2, 4), dtype=torch.long)
    xn = npg.Tensor(x.detach().numpy().astype(np.float32))
    yn = npg.Tensor(target.detach().numpy().astype(np.int32))
    
    logitst = mlpt(x)
    losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), target.view(-1))
    losst.backward()
    
    logitsn = mlpn(xn)
    lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
    lossn.backward()
    
    assert torch.allclose(torch.from_numpy(mlpn.c_fc.weight.grad), mlpt.c_fc.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_fc.bias.grad), mlpt.c_fc.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_proj.weight.grad), mlpt.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_proj.bias.grad), mlpt.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)
    
def test_block_manual():
    config = GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0)
    lnt1 = LayerNormTorch(4, bias=True)
    attnt = CausalSelfAttentionTorch(config)
    lnt2 = LayerNormTorch(4, bias=True)
    mlpt = MLPTorch(config)
    
    lnn1 = npg.nn.LayerNorm(4, bias=True)
    attnn = CausualSelfAttention(config)
    lnn2 = npg.nn.LayerNorm(4, bias=True)
    mlpn = MLP(config)
    
    attnn.c_attn.weight = npg.Tensor(attnt.c_attn.weight.detach().numpy(), requires_grad=True)
    attnn.c_attn.bias = npg.Tensor(attnt.c_attn.bias.detach().numpy(), requires_grad=True)
    attnn.c_proj.weight = npg.Tensor(attnt.c_proj.weight.detach().numpy(), requires_grad=True)
    attnn.c_proj.bias = npg.Tensor(attnt.c_proj.bias.detach().numpy(), requires_grad=True)
    lnn1.weight = npg.Tensor(lnt1.weight.detach().numpy(), requires_grad=True)
    lnn1.bias = npg.Tensor(lnt1.bias.detach().numpy(), requires_grad=True)
    lnn2.weight = npg.Tensor(lnt2.weight.detach().numpy(), requires_grad=True)
    lnn2.bias = npg.Tensor(lnt2.bias.detach().numpy(), requires_grad=True)
    mlpn.c_fc.weight = npg.Tensor(mlpt.c_fc.weight.detach().numpy(), requires_grad=True)
    mlpn.c_fc.bias = npg.Tensor(mlpt.c_fc.bias.detach().numpy(), requires_grad=True)
    mlpn.c_proj.weight = npg.Tensor(mlpt.c_proj.weight.detach().numpy(), requires_grad=True)
    mlpn.c_proj.bias = npg.Tensor(mlpt.c_proj.bias.detach().numpy(), requires_grad=True)
    
    x = torch.randn(2, 4, 4)
    target = torch.randint(0, 4, (2, 4), dtype=torch.long)
    xn = npg.Tensor(x.detach().numpy().astype(np.float32))
    yn = npg.Tensor(target.detach().numpy().astype(np.int32))
    logitst = mlpt(lnt2(x + attnt(lnt1(x))))
    losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), target.view(-1))
    losst.backward()
    
    logitsn = mlpn(lnn2(xn + attnn(lnn1(xn))))
    lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
    lossn.backward()
    
    assert torch.allclose(torch.from_numpy(attnn.c_attn.weight.grad), attnt.c_attn.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(attnn.c_attn.bias.grad), attnt.c_attn.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(attnn.c_proj.weight.grad), attnt.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(attnn.c_proj.bias.grad), attnt.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lnn1.weight.grad), lnt1.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lnn1.bias.grad), lnt1.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lnn2.weight.grad), lnt2.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lnn2.bias.grad), lnt2.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_fc.weight.grad), mlpt.c_fc.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_fc.bias.grad), mlpt.c_fc.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_proj.weight.grad), mlpt.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(mlpn.c_proj.bias.grad), mlpt.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)
    
class test_block():
    config = GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0)
    blockt = BlockTorch(config)
    blockn = Block(config)
    
    blockn.attn.c_attn.weight = npg.Tensor(blockt.attn.c_attn.weight.detach().numpy(), requires_grad=True)
    blockn.attn.c_attn.bias = npg.Tensor(blockt.attn.c_attn.bias.detach().numpy(), requires_grad=True)
    blockn.attn.c_proj.weight = npg.Tensor(blockt.attn.c_proj.weight.detach().numpy(), requires_grad=True)
    blockn.attn.c_proj.bias = npg.Tensor(blockt.attn.c_proj.bias.detach().numpy(), requires_grad=True)
    blockn.ln_1.weight = npg.Tensor(blockt.ln_1.weight.detach().numpy(), requires_grad=True)
    blockn.ln_1.bias = npg.Tensor(blockt.ln_1.bias.detach().numpy(), requires_grad=True)
    blockn.ln_2.weight = npg.Tensor(blockt.ln_2.weight.detach().numpy(), requires_grad=True)
    blockn.ln_2.bias = npg.Tensor(blockt.ln_2.bias.detach().numpy(), requires_grad=True)
    blockn.mlp.c_fc.weight = npg.Tensor(blockt.mlp.c_fc.weight.detach().numpy(), requires_grad=True)
    blockn.mlp.c_fc.bias = npg.Tensor(blockt.mlp.c_fc.bias.detach().numpy(), requires_grad=True)
    blockn.mlp.c_proj.weight = npg.Tensor(blockt.mlp.c_proj.weight.detach().numpy(), requires_grad=True)
    blockn.mlp.c_proj.bias = npg.Tensor(blockt.mlp.c_proj.bias.detach().numpy(), requires_grad=True)
    
    xt = torch.randn(2, 4, 4)
    target = torch.randint(0, 4, (2, 4), dtype=torch.long)
    xn = npg.Tensor(xt.detach().numpy().astype(np.float32))
    yn = npg.Tensor(target.detach().numpy().astype(np.int32))
    
    logitst = blockt(xt)
    losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), target.view(-1))
    losst.backward()
    
    logitsn = blockn(xn)
    lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
    lossn.backward()
    
    assert torch.allclose(torch.from_numpy(blockn.attn.c_attn.weight.grad), blockt.attn.c_attn.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.attn.c_attn.bias.grad), blockt.attn.c_attn.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.attn.c_proj.weight.grad), blockt.attn.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.attn.c_proj.bias.grad), blockt.attn.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.ln_1.weight.grad), blockt.ln_1.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.ln_1.bias.grad), blockt.ln_1.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.ln_2.weight.grad), blockt.ln_2.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.ln_2.bias.grad), blockt.ln_2.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.mlp.c_fc.weight.grad), blockt.mlp.c_fc.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.mlp.c_fc.bias.grad), blockt.mlp.c_fc.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.mlp.c_proj.weight.grad), blockt.mlp.c_proj.weight.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(blockn.mlp.c_proj.bias.grad), blockt.mlp.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
    assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)
