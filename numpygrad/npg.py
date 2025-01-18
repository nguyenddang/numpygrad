'''Base functionality for npg (numpygrad) Module'''

import numpy as np
from numpygrad.engine import Tensor
# utility functions
def exp(x:Tensor) -> Tensor:
    return x.exp()

def log(x:Tensor) -> Tensor:
    return x.log()

def mean(x:Tensor, dim:int=None) -> Tensor:
    axis_size = x.data.shape[dim] if dim is not None else x.data.size
    return x.sum(dim=dim) / axis_size

def sum(x:Tensor, dim:int=None) -> Tensor:
    return x.sum(axis=dim)

# activation functions
def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), _children=(x,), grad_fn='ReLUBackward', requires_grad=x.requires_grad)
    # relu needs _backward defined, special case
    def _backward():
        x.grad += (x.data > 0) * out.grad  # Gradient is 1 where data > 0, else 0
    out._backward = _backward
    return out
def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())

def tanh(x: Tensor) -> Tensor:
    e_x = x.exp()
    ne_x = (-x).exp()
    return (e_x - ne_x) / (e_x + ne_x)

def gelu(x: Tensor) -> Tensor:
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x**3)))

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_shifted = x - Tensor(np.max(x.data, axis=dim, keepdims=True))
    e_x = x_shifted.exp()
    return e_x / e_x.sum(dim=dim, keepdims=True)

def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    prob = softmax(logits, dim=-1)
    class_prob = prob[np.arange(prob.shape[0]), target.data]
    loss = -log(class_prob)
    return mean(loss)

