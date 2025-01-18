import numpy as np 

class Tensor:
    def __init__(self, data, _children=(), grad_fn=None):
        assert isinstance(data, np.ndarray) or isinstance(data, list), f"Expected data to be of type np.ndarray or list, not {type(data)}"
        self.data = data
        self.shape = data.shape
        self._prev = set(_children)
        self.grad = 0
        self._backward = lambda: None
        self.grad_fn = grad_fn
        
    def check_type_add_mul(self, other):
        assert type(other) == Tensor or type(other) == int or type(other) == float, f"Expected other to be of type Tensor, int, or float, not {type(other)}"
        dtype = self.data.dtype
        if type(other) == int or type(other) == float:
            other = Tensor(np.array([other], dtype=dtype))
        return other
    
    def check_broadcast(self, other, out):
        # check if broadcasting is needed, align to the right similar to numpy & torch 
        expanded_other_shape = other.shape
        expanded_data_shape = self.shape
        if len(self.shape) < len(other.shape):
            expanded_data_shape = (1,) * (len(other.shape) - len(self.shape)) + self.shape
        elif len(self.shape) > len(other.shape):
            expanded_other_shape = (1,) * (len(self.shape) - len(other.shape)) + other.shape
            expanded_data_shape = self.shape
        # get axes to sum along else None
        broadcasted_axes_self = tuple(i for i, (a,c) in enumerate(zip(expanded_data_shape, out.shape)) if a == 1 and c > 1)
        broadcasted_axes_other = tuple(i for i, (b,c) in enumerate(zip(expanded_other_shape, out.shape)) if b == 1 and c > 1)
        return broadcasted_axes_self, broadcasted_axes_other
    
    def __add__(self, other):
        other = self.check_type_add_mul(other)
        out =  Tensor(self.data + other.data, _children=(self, other), grad_fn='AddBackward')
        broadcasted_axes_self, broadcasted_axes_other = self.check_broadcast(other, out)
        def _backward():
            sgrad = 1.0 * out.grad
            ograd = 1.0 * out.grad
            self.grad += np.sum(sgrad, axis=broadcasted_axes_self) if broadcasted_axes_self else sgrad
            other.grad += np.sum(ograd, axis=broadcasted_axes_other) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self.check_type_add_mul(other)
        out =  Tensor(self.data * other.data, _children=(self, other), grad_fn='MulBackward')
        broadcasted_axes_self, broadcasted_axes_other = self.check_broadcast(other, out)
        def _backward():
            sgrad = other.data * out.grad
            ograd = self.data * out.grad
            self.grad += np.sum(sgrad, axis=broadcasted_axes_self) if broadcasted_axes_self else sgrad
            other.grad += np.sum(ograd, axis=broadcasted_axes_other) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        assert type(other) == Tensor, f"Expected other to be of type Tensor, not {type(other)}"
        out =  Tensor(self.data @ other.data, _children=(self, other), grad_fn='MatmulBackward')
        broadcasted_axes_self, broadcasted_axes_other = self.check_broadcast(other, out)
        def _backward():
            sgrad = out.grad @ np.swapaxes(other.data, -1, -2)
            ograd = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += np.sum(sgrad, axis=broadcasted_axes_self) if broadcasted_axes_self else sgrad
            other.grad += np.sum(ograd, axis=broadcasted_axes_other) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = self.check_type_add_mul(other)
        out =  Tensor(self.data ** other.data, _children=(self,), grad_fn='PowBackward')
        def _backward():
            self.grad += other.data * self.data ** (other.data-1) * out.grad
        out._backward = _backward
        return out

    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), _children=(self,), grad_fn='SigmoidBackward')
        def backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = backward
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), _children=(self,), grad_fn='TanhBackward')
        def backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), grad_fn='ReluBackward')
        def backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = backward
        return out
    
    def sum(self,axis=None):
        sum = np.sum(self.data, axis=axis)
        out = Tensor(np.array(sum), _children=(self,), grad_fn='SumBackward')
        def backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = backward
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
        
    
        
    
    