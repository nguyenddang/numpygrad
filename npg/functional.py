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

def mean(x:npg.Tensor, dim:int=None, keepdims=False) -> npg.Tensor:
    axis_size = x.data.shape[dim] if dim is not None else x.data.size
    return x.sum(dim=dim, keepdims=keepdims) / axis_size

def var(x:npg.Tensor, dim:int=None, keepdims=False) -> npg.Tensor:
    mean_x = mean(x, dim, keepdims=keepdims)
    return mean((x - mean_x)**2, dim, keepdims=keepdims)

def sum(x:npg.Tensor, dim:int=None) -> npg.Tensor:
    return x.sum(dim=dim)

def randn(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

def rand(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

def zeros(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.zeros(*shape).astype(dtype), requires_grad=requires_grad)

def ones(*shape, requires_grad=False, dtype=np.float32) -> npg.Tensor:
    return npg.Tensor(np.ones(*shape).astype(dtype), requires_grad=requires_grad)

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
    return 1 / (1 + (-x).exp())

def tanh(x: npg.Tensor) -> npg.Tensor:
    e_x = x.exp()
    ne_x = (-x).exp()
    return (e_x - ne_x) / (e_x + ne_x)

def gelu(x: npg.Tensor) -> npg.Tensor:
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x**3)))

def softmax(x: npg.Tensor, dim: int = -1) -> npg.Tensor:
    x_shifted = x - npg.Tensor(np.max(x.data, axis=dim, keepdims=True))
    e_x = x_shifted.exp()
    return e_x / e_x.sum(dim=dim, keepdims=True)

# loss functions
def cross_entropy(logits: npg.Tensor, target: npg.Tensor) -> npg.Tensor:
    prob = softmax(logits, dim=-1)
    class_prob = prob[np.arange(prob.shape[0]), target.data]
    loss = -log(class_prob)
    return mean(loss)

def mse_loss(pred: npg.Tensor, target: npg.Tensor) -> npg.Tensor:
    return mean((pred - target)**2)


