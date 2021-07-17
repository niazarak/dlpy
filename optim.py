from typing import List, Tuple

import numpy as np


class SGD:
    def __init__(
            self,
            lr: float,
            parameters: List[np.ndarray],
            grads: List[np.ndarray],
            ridge_lambda: float = 0.0
    ):
        self.lr = lr
        self.ridge_lambda = ridge_lambda
        self.parameters = parameters
        self.grads = grads

    def step(self):
        for param, grad in zip(self.parameters, self.grads):
            diff = grad * self.lr
            if self.ridge_lambda:
                diff += 2 * param * self.ridge_lambda
            param[:] = param - diff

    def zerograd(self):
        for grad in self.grads:
            grad[:] = np.zeros_like(grad)


class Adam:
    def __init__(
            self,
            lr: float,
            parameters: List[np.ndarray],
            grads: List[np.ndarray],
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
    ):
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.t_betas = betas

        self.parameters = parameters

        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]

        self.grads = grads

    def step(self):
        for p_i, param, grad in zip(range(len(self.parameters)), self.parameters, self.grads):
            self.m[p_i] = self.betas[0] * self.m[p_i] + (1 - self.betas[0]) * grad
            self.v[p_i] = self.betas[1] * self.v[p_i] + (1 - self.betas[1]) * (grad ** 2)
            m = self.m[p_i] / (1 - self.t_betas[0])
            v = self.v[p_i] / (1 - self.t_betas[1])

            diff = self.lr * (m / (v ** 0.5 + self.eps))
            param[:] = param - diff

        self.t_betas = (self.t_betas[0] * self.betas[0], self.t_betas[1] * self.betas[1])

    def zerograd(self):
        for grad in self.grads:
            grad[:] = np.zeros_like(grad)
