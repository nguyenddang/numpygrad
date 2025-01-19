import npg 

class Module:
    
    def __init__(self):
        self.training = True
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
            
    def parameters(self):
        return []
    
    def __call__(self, *inputs):
        return self.forward(*inputs)

# Linear
class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = npg.randn(in_features, out_features, requires_grad=True)
        self.bias = npg.randn(out_features, requires_grad=True) if bias else None
        
    def forward(self, x):
        return x @ self.weight + self.bias
    
    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]
    
    def __repr__(self):
        return f"npg.nn.Linear({self.weight.data.shape[0]}, {self.weight.data.shape[1]})"

# Activation functions
class ReLU(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return npg.relu(x)
    
    def __repr__(self):
        return "npg.nn.ReLU()"
    
class Sigmoid(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return npg.sigmoid(x)
    
    def __repr__(self):
        return "npg.nn.Sigmoid()"
    
class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return npg.tanh(x)
    
    def __repr__(self):
        return "npg.nn.Tanh()"
    
class GeLU(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return npg.gelu(x)
    
    def __repr__(self):
        return "npg.nn.GeLU()"
    
# Normalization
class LayerNorm(Module):
    
    def __init__(self, ndim, eps=1e-5, bias=True):
        super().__init__()
        self.weight = npg.ones(ndim, requires_grad=True)
        self.bias = npg.zeros(ndim, requires_grad=True) if bias else None
        self.eps = eps
    
    def forward(self, x):
        mean = npg.mean(x, dim=-1, keepdim=True)
        var = npg.var(x, dim=-1, keepdim=True)
        norm = (x - mean) / npg.sqrt(var + self.eps)
        out = norm * self.weight + self.bias
        return out
    
    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]

    def __repr__(self):
        return f"npg.nn.LayerNorm({self.weight.data.shape[0]})"
    
# Regularisation
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1, "Dropout probability must be in the range [0, 1)."
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (npg.rand(*x.data.shape) > self.p)
            return x * mask / (1 - self.p)
        return x
    
    def __repr__(self):
        return f"npg.nn.Dropout(p={self.p})"

# Loss functions
        
        

    

    
        
    