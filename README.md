# Numpygrad: npg

A tiny autograd engine with ~1% of Pytorch's functionality built purely from Numpy. Some functionality:
-   `npg.nn`: ~ `torch.nn` (but currently only include basic ones)
- `npg.functional`: ~ `torch.nn.functional` (again very very basics ones only)
- `npg.engine`: autograd
- `npg.optim`: just AdamW for now...

This repo is a decent starting point to understand the internal of autograd engines and why Pytorch is so great:). To get started or build on top of.
```bash
git clone https://github.com/nguyenddang/numpygrad.git
pip install numpy
pip install torch       # optional, needed to run tests
pip install pytest      # for testing
pip install -e .        # install npg as a library so you only need to do import npg
```

# Example
Most functions are identical to `torch`, for example:
```python
import npg

# basic ops
A = npg.randn(2,3)
A_mean = npg.mean(A, dim=-1)
A_var = npg.var(A, dim=0)
A_sum = npg.sum(A)

ones = npg.ones(2,3)
A_ones = np.ones_like(A)
```

`npg` includes `nn.Module` as well so we can build a fully functional neural net classes. Here is `CausalSelfAttention`:
```python
import npg
import npg.nn as nn

class CausualSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.mask = npg.tril(npg.ones((1, 1, config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)
    
    def forward(self, x: npg.Tensor):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.reshape(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        # attention 
        att = (q @ k.transpose(-2, -1)) * (1.0/ k.shape[-1]**0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -np.inf)
        att = npg.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y
```

In the `demo/nanogpt` folder you will model definition (`model.py`) and training code (`train.py`) to train a character-NanoGPT on the mini-Shakespere dataset written completely with `npg`. To try it out, run:
```python
python demo/nanogpt/data/prepare.py # download min-shakespere dataset
python demo/nanogpt/train.py # train nanogpt using npg
```  
Some prints during training...
```
iter 95, loss: 2.663806 time: 546.59ms
iter 96, loss: 2.648255 time: 579.21ms
iter 97, loss: 2.563105 time: 580.07ms
iter 98, loss: 2.676012 time: 618.23ms
iter 99, loss: 2.479903 time: 583.02ms
iter 100, loss: 2.700533 time: 579.66ms
train loss: 2.6035077571868896, val loss: 2.6220905780792236
iter 101, loss: 2.609036 time: 4788.23ms
iter 102, loss: 2.642269 time: 556.65ms
iter 103, loss: 2.525060 time: 510.07ms
iter 104, loss: 2.631540 time: 595.66ms
iter 105, loss: 2.614602 time: 561.78ms
iter 106, loss: 2.657294 time: 587.83ms
iter 107, loss: 2.551862 time: 590.85ms
iter 108, loss: 2.621282 time: 549.98ms
iter 109, loss: 2.660579 time: 558.85ms
iter 110, loss: 2.557863 time: 608.98ms
```
So ... it might take a while but hey it trains!!! Running the same model with Pytorch on CPU (same config everything) takes ~`30ms` per iter so we're about 20x slower:) 

# Run tests

Under `test/` are extensive testing code for comparing gradients calculation between `npg` and `torch`. To run tests:
```bash
python -m pytest
```


Enjoy!