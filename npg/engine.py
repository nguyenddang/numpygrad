import numpy as np 

class Tensor:
    def __init__(self, data, _children=(), grad_fn=None, requires_grad=False):
        assert isinstance(data, np.ndarray) or isinstance(data, list), f"Expected data to be of type np.ndarray or list, not {type(data)}"
        self.data = np.asarray(data) # make sure data is a numpy array
        self.shape = data.shape 
        self._prev = set(_children)
        self.grad = None
        self._backward = lambda: None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        self.dtype = data.dtype
        
    def check_op(self, other):
        if not isinstance(other, Tensor):
            # convert to tensor
            other = Tensor(np.asarray(other), requires_grad=False)
        out_requires_grad = self.requires_grad or other.requires_grad
        return other, out_requires_grad
    
    def check_broadcast(self, other, out):
        def expand_shape(shape, target_len):
            # expand shape to target size
            return (1,) * (target_len - len(shape)) + shape
        expanded_self_shape = expand_shape(self.shape, len(out.shape))
        expanded_other_shape = expand_shape(other.shape, len(out.shape))
        # find axes that need to be broadcasted
        broadcasted_axes_self = tuple(i for i, (a, c) in enumerate(zip(expanded_self_shape, out.shape)) if a == 1 and c > 1)
        broadcasted_axes_other = tuple(i for i, (b, c) in enumerate(zip(expanded_other_shape, out.shape)) if b == 1 and c > 1)
        keepdim_self = (len(self.shape) == len(other.shape)) and (len(broadcasted_axes_self)) > 0
        keepdim_other = (len(self.shape) == len(other.shape)) and (len(broadcasted_axes_other)) > 0
        return broadcasted_axes_self, broadcasted_axes_other, keepdim_self, keepdim_other
    
    @property
    def grad(self):
        if self._grad is None and self.requires_grad:
            self._grad = np.zeros_like(self.data, dtype=np.float32)
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value
    
    def __add__(self, other):
        other, out_requires_grad = self.check_op(other)
        out = Tensor(self.data + other.data, _children=(self, other), grad_fn='AddBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdim_self, keepdim_other = self.check_broadcast(other, out)
        def _backward():
            if self.requires_grad:
                sgrad = np.sum(out.grad, axis=broadcasted_axes_self, keepdims=keepdim_self) if broadcasted_axes_self else out.grad
                self.grad += sgrad
            if other.requires_grad:
                ograd = np.sum(out.grad, axis=broadcasted_axes_other, keepdims=keepdim_other) if broadcasted_axes_other else out.grad
                other.grad += ograd
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other, out_requires_grad = self.check_op(other)
        out = Tensor(self.data * other.data, _children=(self, other), grad_fn='MulBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdim_self, keepdim_other = self.check_broadcast(other, out)
        
        def _backward():
            if self.requires_grad:
                grad_self = np.sum(other.data * out.grad, axis=broadcasted_axes_self, keepdims=keepdim_self) if broadcasted_axes_self else other.data * out.grad
                self.grad += grad_self
            if other.requires_grad:
                grad_other = np.sum(self.data * out.grad, axis=broadcasted_axes_other, keepdims=keepdim_other) if broadcasted_axes_other else self.data * out.grad
                other.grad += grad_other
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other, out_requires_grad = self.check_op(other)
        out =  Tensor(self.data @ other.data, _children=(self, other), grad_fn='MatmulBackward', requires_grad=out_requires_grad)
        broadcasted_axes_self, broadcasted_axes_other, keepdim_self, keepdim_other = self.check_broadcast(other, out)
        def _backward():
            if self.requires_grad:
                sgrad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += np.sum(sgrad, axis=broadcasted_axes_self, keepdims=keepdim_self) if broadcasted_axes_self else sgrad
            if other.requires_grad:
                ograd = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += np.sum(ograd, axis=broadcasted_axes_other, keepdims=keepdim_other) if broadcasted_axes_other else ograd
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, int) or isinstance(other, float), f"Expected other to be of type int or float, not {type(other)}"
        out =  Tensor(self.data ** other, _children=(self,), grad_fn='PowBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                sgrad = other * self.data ** (other - 1) * out.grad
                self.grad += sgrad
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
    
    def sum(self, dim=None, keepdim=False):
        sum_data = np.sum(self.data, axis=dim, keepdims=keepdim)
        sum_data = np.array(sum_data) # make sure data is a numpy array
        out = Tensor(sum_data, _children=(self,), grad_fn='SumBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                outgrad = np.expand_dims(out.grad, axis=dim) if dim is not None and not keepdim else out.grad
                self.grad += np.ones_like(self.data) * outgrad
        out._backward = _backward
        return out   
    
    def reshape(self, *shape):
        new_data = self.data.reshape(shape)
        out = Tensor(new_data, _children=(self,), grad_fn='ReshapeBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += np.reshape(out.grad, self.data.shape)
        out._backward = _backward
        return out
    
    def split(self, split_size, dim=0):
        assert self.shape[dim] % split_size == 0, f"Cannot split tensor of size {self.shape[dim]} by {split_size}"
        num_splits = self.shape[dim] // split_size
        split_data = np.split(self.data, num_splits, axis=dim)

        split_tensors = []
        for i, data in enumerate(split_data):
            split_tensor = Tensor(data, _children=(self,), grad_fn='SplitBackward', requires_grad=self.requires_grad)
            
            def _backward(split_tensor=split_tensor, start_idx=i * split_size, end_idx=(i + 1) * split_size):
                if self.requires_grad:
                    # Create a gradient slice for the original tensor
                    grad_slice = [slice(None)] * len(self.shape)
                    grad_slice[dim] = slice(start_idx, end_idx)
                    self.grad[tuple(grad_slice)] += split_tensor.grad

            split_tensor._backward = _backward
            split_tensors.append(split_tensor)

        return split_tensors
    
    def transpose(self, dim0, dim1):
        transposed_data = np.swapaxes(self.data, dim0, dim1)
        out = Tensor(transposed_data, _children=(self,), grad_fn='TransposeBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, dim0, dim1)
        out._backward = _backward
        return out
    
    def masked_fill(self, mask, value: float):
        assert mask.dtype == np.bool_, f"Expected mask to be of type np.bool, not {mask.dtype}"
        mask = np.broadcast_to(mask.data, self.shape)
        mask_data = np.where(mask, value, self.data)
        out = Tensor(mask_data, _children=(self,), grad_fn='MaskedFillBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                self.grad += (~mask).astype(np.float32) * out.grad
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
    
    def __eq__(self, other):
        # compare data with int or float
        return Tensor(self.data == other, requires_grad=False)
    
    def __gt__(self, other):
        return Tensor(self.data > other, requires_grad=False)
    
    def __lt__(self, other):
        return Tensor(self.data < other, requires_grad=False)
    
    def __ge__(self, other):
        return Tensor(self.data >= other, requires_grad=False)
    
    def __le__(self, other):
        return Tensor(self.data <= other, requires_grad=False)
        
    def __hash__(self):
        return id(self)  # Use the object's ID as a hash value
    
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
        if isinstance(idx, Tensor):
            idx = idx.data
        sliced_data = self.data[idx]
        out = Tensor(sliced_data, _children=(self,), grad_fn='IndexBackward', requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out
    
    def __setitem__(self, idx, value):
        if isinstance(idx, Tensor):
            idx = idx.data
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = value
    
    def item(self,):
        assert self.data.size == 1, "Only tensors with one element can be converted to Python scalars"
        return self.data.item()
        
        
    
        
    
    