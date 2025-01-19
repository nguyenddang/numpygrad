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
    

        
        

