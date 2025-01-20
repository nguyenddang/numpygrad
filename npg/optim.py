import numpy as np

class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        # Perform a single optimization step
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                p_update += self.weight_decay * param.data
            param.data -= p_update

    def zero_grad(self):
        for param in self.params:
            param.grad = None