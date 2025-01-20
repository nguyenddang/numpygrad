'''Base functionality for npg (numpygrad) Module'''

import numpy as np
import npg
# utility functions
def exp(x:npg.Tensor) -> npg.Tensor:
    return x.exp()

def log(x:npg.Tensor) -> npg.Tensor:
    return x.log()

def sqrt(x:npg.Tensor) -> npg.Tensor:
    return x**0.5

def mean(x:npg.Tensor, dim:int=None, keepdim=False) -> npg.Tensor:
    mean_data = np.mean(x.data, axis=dim, keepdims=keepdim)
    mean_data = np.array(mean_data)
    out = npg.Tensor(mean_data, _children=(x,), grad_fn='MeanBackward', requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            scale = np.prod(x.data.shape) if dim is None else x.data.shape[dim]
            outgrad = np.expand_dims(out.grad, axis=dim) if dim is not None and not keepdim else out.grad
            x.grad += outgrad * np.ones_like(x.data) / scale
    out._backward = _backward
    return out
            
def var(x: npg.Tensor, dim: int = None, keepdim=False) -> npg.Tensor:
    mean_data = np.mean(x.data, axis=dim, keepdims=keepdim)
    var_data = np.mean((x.data - mean_data) ** 2, axis=dim, keepdims=keepdim)
    var_data = np.array(var_data)
    out = npg.Tensor(var_data, _children=(x,), grad_fn='VarBackward', requires_grad=x.requires_grad)
    def _backward():
        if x.requires_grad:
            scale = np.prod(x.data.shape) if dim is None else x.data.shape[dim]
            grad = 2 * (x.data - mean_data) / scale
            outgrad = np.expand_dims(out.grad, axis=dim) if dim is not None and not keepdim else out.grad
            x.grad += grad * outgrad
    out._backward = _backward
    return out

def sum(x:npg.Tensor, dim:int=None) -> npg.Tensor:
    return x.sum(dim=dim)

def randn(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

def rand(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

def randint(low, high=None, size=None, requires_grad=False, dtype=np.int32) -> npg.Tensor:
    return npg.Tensor(np.random.randint(low, high, size).astype(dtype), requires_grad=requires_grad)

def zeros(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.zeros(*shape).astype(dtype), requires_grad=requires_grad)

def zeros_like(x:npg.Tensor) -> npg.Tensor:
    return npg.Tensor(np.zeros_like(x.data), requires_grad=x.requires_grad)

def ones(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.ones(*shape).astype(dtype), requires_grad=requires_grad)

def ones_like(x:npg.Tensor) -> npg.Tensor:
    return npg.Tensor(np.ones_like(x.data), requires_grad=x.requires_grad)

def tril(x:npg.Tensor) -> npg.Tensor:
    return npg.Tensor(np.tril(x.data), _children=(x,))

# activation functions
def relu(x: npg.Tensor) -> npg.Tensor:
    out = npg.Tensor(np.maximum(0, x.data), _children=(x,), grad_fn='ReLUBackward', requires_grad=x.requires_grad)
    # relu needs _backward defined, special case
    def _backward():
        x.grad += (x.data > 0) * out.grad  # Gradient is 1 where data > 0, else 0
    out._backward = _backward
    return out

def sigmoid(x: npg.Tensor) -> npg.Tensor:
    sigmoid = 1 / (1 + np.exp(-x.data))
    out = npg.Tensor(sigmoid, _children=(x,), grad_fn='SigmoidBackward', requires_grad=x.requires_grad)
    def _backward():
        if x.requires_grad:
            x.grad += sigmoid * (1 - sigmoid) * out.grad
    out._backward = _backward
    return out

def tanh(x: npg.Tensor) -> npg.Tensor:
    tanh = np.tanh(x.data)
    out = npg.Tensor(tanh, _children=(x,), grad_fn='TanhBackward', requires_grad=x.requires_grad)
    def _backward():
        if x.requires_grad:
            x.grad += (1 - tanh**2) * out.grad
    out._backward = _backward
    return out

def gelu(x: npg.Tensor) -> npg.Tensor:
    gelu = 0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3)))
    out = npg.Tensor(gelu, _children=(x,), grad_fn='GELUBackward', requires_grad=x.requires_grad)
    def _backward():
        if x.requires_grad:
            tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3))
            grad = 0.5 * (1 + tanh_out) + 0.5 * x.data * (1 - tanh_out**2) * (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x.data**2))
            x.grad += grad * out.grad

    out._backward = _backward
    return out

def softmax(x: npg.Tensor, dim: int = -1) -> npg.Tensor:
    max_vals = np.max(x.data, axis=dim, keepdims=True)  # shift for stability
    exp_data = np.exp(x.data - max_vals)
    softmax = exp_data / np.sum(exp_data, axis=dim, keepdims=True) 
    out = npg.Tensor(softmax, _children=(x,), grad_fn='SoftmaxBackward', requires_grad=x.requires_grad)
    def _backward():
        if x.requires_grad:
            softmax_x = out.data
            grad = softmax_x * (out.grad - np.sum(out.grad * softmax_x, axis=dim, keepdims=True))
            x.grad += grad
    out._backward = _backward
    return out
 
# loss functions
def cross_entropy(x: npg.Tensor, target: npg.Tensor) -> npg.Tensor:
    max_vals = np.max(x.data, axis=-1, keepdims=True)  # shift for stability
    exp_data = np.exp(x.data - max_vals)
    probs = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
    loss = -np.log(probs[np.arange(x.data.shape[0]), target.data]).mean()
    out = npg.Tensor(np.array(loss), _children=(x, target), grad_fn='CrossEntropyBackward', requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            softmax_x = probs
            softmax_x[np.arange(x.data.shape[0]), target.data] -= 1
            x.grad += softmax_x * out.grad / x.data.shape[0]
    out._backward = _backward
    return out

def mse_loss(pred: npg.Tensor, target: npg.Tensor) -> npg.Tensor:
    return npg.mean((pred - target)**2)


