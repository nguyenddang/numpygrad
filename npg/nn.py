import npg 

class Module:
    def __init__(self):
        self.training = True
        self._modules = {}
        self._parameters = []

    def forward(self, *inputs):
        raise NotImplementedError

    def parameters(self):
        params = self._parameters.copy()
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_module(self, name, module):
        self._modules[name] = module

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, npg.Tensor) and value.requires_grad:
            self._parameters.append(value)
        super().__setattr__(name, value)

    def __repr__(self):
        # Generate a string representation of the module and its submodules
        module_str = self.__class__.__name__ + '(\n'
        for name, module in self._modules.items():
            module_str += f'  ({name}): {add_indent(repr(module), 2)}\n'
        module_str += ')'
        return module_str
    
    def apply(self, fn):
        fn(self)
        for module in self._modules.values():
            module.apply(fn)
        return self

# Linear
class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = npg.randn(out_features, in_features, requires_grad=True)
        self.bias = npg.randn(out_features, requires_grad=True) if bias else None
        
    def forward(self, x):
        return x @ self.weight.transpose(0, 1) + self.bias
    
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

# Embedding
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = npg.randn(num_embeddings, embedding_dim, requires_grad=True)
        
    def forward(self, x):
        return self.weight[x]
    
    def __repr__(self):
        return f"npg.nn.Embedding({self.weight.data.shape[0]}, {self.weight.data.shape[1]})"
    
# others
class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is None:
            modules = []
        self.modules = modules
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def append(self, module):
        self.modules.append(module)
        self.add_module(str(len(self.modules) - 1), module)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def forward(self, *inputs):
        raise NotImplementedError("ModuleList does not implement forward method")

    def __repr__(self):
        # Generate a string representation of the ModuleList and its modules
        module_str = self.__class__.__name__ + '(\n'
        for idx, module in enumerate(self.modules):
            module_str += f'  ({idx}): {add_indent(repr(module), 2)}\n'
        module_str += ')'
        return module_str
    
def add_indent(s_, num_spaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s
        
        

    

    
        
    