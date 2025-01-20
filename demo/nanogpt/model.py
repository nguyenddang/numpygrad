'''Reference Andrej Karpathy's nanogpt https://github.com/karpathy/nanoGPT'''

import npg
import npg.nn as nn # don't mistake this for torch.nn:)
from dataclasses import dataclass
import numpy as np

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

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GeLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: npg.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x: npg.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPTConfig:
    # config for nanogpt shakespeare
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()}")

    def get_num_params(self):
        return sum([np.prod(p.data.shape) for p in self.parameters()])

    def forward(self, x: npg.Tensor, target: npg.Tensor):
        b, t = x.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = np.arange(0, t, dtype=np.int32)
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        if target is not None:
            logits = self.lm_head(x)
            loss = npg.cross_entropy(logits.reshape(-1, 65), target.reshape(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = np.random.normal(0.0, 0.02, module.weight.data.shape).astype(np.float32)
            if module.bias is not None:
                module.bias.data = np.zeros_like(module.bias.data)
        elif isinstance(module, nn.Embedding):
            module.weight.data = np.random.normal(0.0, 0.02, module.weight.data.shape).astype(np.float32)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data = np.zeros_like(module.bias.data)
            module.weight.data = np.ones_like(module.weight.data)
        
        

