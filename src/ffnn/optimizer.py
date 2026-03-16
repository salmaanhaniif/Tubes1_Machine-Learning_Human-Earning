# optimizer.py
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # first moment
        self.v = {}  # second moment

    def update(self, params):
        self.t += 1
        for i, p in enumerate(params):
            if i not in self.m:
                self.m[i] = np.zeros_like(p.data)
                self.v[i] = np.zeros_like(p.data)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            p.grad = np.zeros_like(p.data)