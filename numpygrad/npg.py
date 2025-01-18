'''Base functionality for npg (numpygrad) Module'''

import numpy as np
from engine import Tensor

# activation functions
def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), _children=(x,), grad_fn='ReLUBackward')
    # relu needs _backward defined, special case
    def _backward():
        x.grad += (x.data > 0) * out.grad  # Gradient is 1 where data > 0, else 0
    
    out._backward = _backward
    return out

def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())

def tanh(x: Tensor) -> Tensor:
    return (x - x.exp()) / (x + x.exp())

def gelu(x: Tensor) -> Tensor:
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x**3)))

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    e_x = (x - np.max(x, axis=-1, keepdims=True)).exp()
    return e_x / e_x.sum(axis=axis, keepdims=True)

