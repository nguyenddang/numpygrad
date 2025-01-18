import npg 

class Module:
    
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
        self.W = npg.randn(in_features, out_features, requires_grad=True)
        self.b = npg.randn(out_features, requires_grad=True) if bias else None
        
    def forward(self, x):
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b] if self.b is not None else [self.W]
    
    def __repr__(self):
        return f"npg.nn.Linear({self.W.data.shape[0]}, {self.W.data.shape[1]})"

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
class Layernorm(Module):
    
    def __init__(self, normalized_shape, eps=1e-5, bias=True):
        super().__init__()
        self.weight = npg.ones(normalized_shape, requires_grad=True)
        self.bias = npg.zeros(normalized_shape, requires_grad=True) if bias else None
        self.eps = eps
    
    def forward(self, x):
        xmean = npg.mean(x, dim=-1, keepdims=True)
        xvar = npg.var(x, dim=-1, keepdims=True)
        xhat = (x - xmean) / npg.sqrt(xvar + self.eps)
        out = xhat * self.weight + self.bias
        return out
    
    def parameters(self):
        return [self.weight, self.bias] if self.beta is not None else [self.weight]
        
        

    

    
        
    