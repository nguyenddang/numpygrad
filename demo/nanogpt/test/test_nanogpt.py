# import npg
# import torch
# import numpy as np
# from demo.nanogpt.model import CausualSelfAttention
# from model_torch import CausalSelfAttention as CausalSelfAttentionTorch, GPTConfig

# def test_attention():
#     config = GPTConfig(block_size=4, n_embd=4, n_head=2, n_layer=1, dropout=0.0)
#     blockt = CausalSelfAttentionTorch(config)
#     blockn = CausualSelfAttention(config)
    
#     xt = torch.randn(1, 4, 4)
#     yt = torch.randint(0, 4, (1, 4), dtype=torch.long)
#     xn = npg.Tensor(xt.detach().numpy().astype(np.float32))
#     yn = npg.Tensor(yt.detach().numpy().astype(np.int32))
    
#     logitst = blockt(xt)
#     losst = torch.nn.functional.cross_entropy(logitst.view(-1, 4), yt.view(-1))
#     losst.backward()
    
#     logitsn = blockn(xn)
#     lossn = npg.cross_entropy(logitsn.reshape(-1, 4), yn.reshape(-1))
#     lossn.backward()
    
#     assert torch.allclose(torch.from_numpy(blockn.c_attn.W.grad), blockt.c_attn.weight.grad.T, rtol=1e-5, atol=1e-6)
#     assert torch.allclose(torch.from_numpy(blockn.c_attn.b.grad), blockt.c_attn.bias.grad, rtol=1e-5, atol=1e-6)
#     assert torch.allclose(torch.from_numpy(blockn.c_proj.W.grad), blockt.c_proj.weight.grad.T, rtol=1e-5, atol=1e-6)
#     assert torch.allclose(torch.from_numpy(blockn.c_proj.b.grad), blockt.c_proj.bias.grad, rtol=1e-5, atol=1e-6)
#     assert torch.allclose(torch.from_numpy(lossn.data), losst.data, rtol=1e-5, atol=1e-6)
#     assert torch.allclose(torch.from_numpy(logitsn.data), logitst.data, rtol=1e-5, atol=1e-6)
    

    
