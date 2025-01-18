import numpy as np 

class Tensor:
    def __init__(self, data, _children=(), grad_fn=None, requires_grad=False):
        assert isinstance(data, np.ndarray) or isinstance(data, list), f"Expected data to be of type np.ndarray or list, not {type(data)}"
        self.data = data
        self.shape = data.shape
        self._prev = set(_children)
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._backward = lambda: None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        
    def check_type_add_mul(self, other):
        intcheck = isinstance(other, int)
        floatcheck = isinstance(other, float)
        tensorcheck = isinstance(other, Tensor)
        assert intcheck or floatcheck or tensorcheck, f"Expected other to be of type Tensor, int, or float, not {type(other)}"
        dtype = self.data.dtype
        if intcheck or floatcheck:
            other = Tensor(np.array([other], dtype=dtype))
        return other
    
    def check_broadcast(self, other, out):
        # check if broadcasting is needed, align to the right similar to numpy & torch 
        expanded_other_shape = other.shape
        expanded_data_shape = self.shape
        keepdims = False
        if len(self.shape) < len(other.shape):
            expanded_data_shape = (1,) * (len(other.shape) - len(self.shape)) + self.shape
            expanded_other_shape = other.shape
        elif len(self.shape) > len(other.shape):
            expanded_other_shape = (1,) * (len(self.shape) - len(other.shape)) + other.shape
            expanded_data_shape = self.shape
        else:
            keepdims = True 
        # get axes to sum along else None
        broadcasted_axes_self = tuple(i for i, (a,c) in enumerate(zip(expanded_data_shape, out.shape)) if a == 1 and c > 1)
        broadcasted_axes_other = tuple(i for i, (b,c) in enumerate(zip(expanded_other_shape, out.shape)) if b == 1 and c > 1)
        return broadcasted_axes_self, broadcasted_axes_other, keepdims
    
    def __add__(self, other):
        other = self.check_type_add_mul(other)
        out_requires_grad = self.requires_grad or other.requires_grad
        out =  Tensor(self.data + other.data, _children=(self, other), grad_fn='AddBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdims = self.check_broadcast(other, out)
        def _backward():
            if self.requires_grad:
                sgrad = 1.0 * out.grad
                self.grad += np.sum(sgrad, axis=broadcasted_axes_self, keepdims=keepdims) if broadcasted_axes_self else sgrad
            if other.requires_grad:
                ograd = 1.0 * out.grad
                other.grad += np.sum(ograd, axis=broadcasted_axes_other, keepdims=keepdims) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self.check_type_add_mul(other)
        out_requires_grad = self.requires_grad or other.requires_grad
        out =  Tensor(self.data * other.data, _children=(self, other), grad_fn='MulBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdims = self.check_broadcast(other, out)
        def _backward():
            if self.requires_grad:
                sgrad = other.data * out.grad
                self.grad += np.sum(sgrad, axis=broadcasted_axes_self, keepdims=keepdims) if broadcasted_axes_self else sgrad
            if other.requires_grad:
                ograd = self.data * out.grad
                other.grad += np.sum(ograd, axis=broadcasted_axes_other, keepdims=keepdims) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        assert type(other) == Tensor, f"Expected other to be of type Tensor, not {type(other)}"
        out_requires_grad = self.requires_grad or other.requires_grad
        out =  Tensor(self.data @ other.data, _children=(self, other), grad_fn='MatmulBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdims = self.check_broadcast(other, out)
        def _backward():
            if self.requires_grad:
                sgrad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += np.sum(sgrad, axis=broadcasted_axes_self,keepdims=keepdims) if broadcasted_axes_self else sgrad
            if other.requires_grad:
                ograd = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += np.sum(ograd, axis=broadcasted_axes_other,keepdims=keepdims) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = self.check_type_add_mul(other)
        out_requires_grad = self.requires_grad or other.requires_grad
        out =  Tensor(self.data ** other.data, _children=(self,), grad_fn='PowBackward', requires_grad=out_requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += other.data * self.data ** (other.data-1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out =  Tensor(np.exp(self.data), _children=(self,), grad_fn='ExpBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += np.exp(self.data) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out =  Tensor(np.log(self.data), _children=(self,), grad_fn='LogBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += 1 / self.data * out.grad
        out._backward = _backward
        return out
    
    def sum(self, dim=None, keepdims=False):
        sum = np.sum(self.data, axis=dim, keepdims=keepdims)
        sum = sum if dim else np.array([sum])
        out =  Tensor(sum, _children=(self,), grad_fn='SumBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                if dim is not None:
                    outgrad = np.expand_dims(out.grad, axis=dim) if out.grad.ndim < self.data.ndim else out.grad
                else:
                    outgrad = out.grad
                self.grad += np.ones_like(self.data) * outgrad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1.0
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + -1.0 * self
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def backward(self):
        # sort children in topological order
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients.")
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # chain rule here
        self.grad = np.ones_like(self.data).astype(np.float32) # set grad of this node to 1s
        for v in reversed(topo):
            v._backward()
            
    def __repr__(self):
        data_str = np.array2string(self.data, separator=', ', prefix='tensor(', suffix=')', precision=4) # makes formatting nicer
        return 'tensor(' + data_str + ', grad_fn=' + str(self.grad_fn) + ')'
    
    def __getitem__(self, idx):
        sliced_data = self.data[idx]
        out = Tensor(sliced_data, _children=(self,), grad_fn='IndexBackward', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                if isinstance(idx, tuple):
                    np.add.at(self.grad, idx, out.grad)
                else:
                    self.grad[idx] += out.grad
        out._backward = _backward
        return out
        
    
        
    
    