import numpy as np

from .abstractions import AbstractOptimizer


class SGD(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 momentum: float = 0.0,
                 nesterov: bool = False):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = 0.0

    def __call__(self, gradient_vector: np.ndarray):
        adjustment = - self.learning_rate * gradient_vector

        if self.momentum:
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_vector
            adjustment = self.velocity

            if self.nesterov:
                adjustment = self.momentum * self.velocity - self.learning_rate * gradient_vector

        return adjustment.copy()


class RMSprop(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 rho: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-7,
                 centered: bool = False):

        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

        self.previous = 0.0
        self.rms = 0.0
        self.mg = 0.0

    def __call__(self, gradient_vector: np.ndarray) -> np.ndarray:
        self.rms = self.rho * self.rms + (1 - self.rho) * gradient_vector ** 2
        direction = self.rms

        if self.centered:
            self.mg = self.rho * self.mg + (1 - self.rho) * gradient_vector
            direction = self.rms - self.mg ** 2

        adjustment = - self.learning_rate * gradient_vector / (np.sqrt(direction) + self.epsilon)

        if self.momentum:
            self.previous = self.momentum * self.previous + adjustment
            adjustment = self.previous

        return adjustment.copy()


class Adam(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 amsgrad: bool = False):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        self.m_t = 0.0
        self.v_t = 0.0
        self.v_hat_t = 0.0

        self.powered_beta_1 = 1.0
        self.powered_beta_2 = 1.0

    def __call__(self, gradient_vector: np.ndarray):
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * gradient_vector
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * gradient_vector ** 2
        direction = self.v_t

        if self.amsgrad:
            self.v_hat_t = np.maximum(self.v_hat_t, self.v_t)
            direction = self.v_hat_t

        self.powered_beta_1 *= self.beta_1
        self.powered_beta_2 *= self.beta_2

        lr_t = self.learning_rate * np.sqrt(1 - self.powered_beta_2) / (1 - self.powered_beta_1)

        return - lr_t * self.m_t / (np.sqrt(direction) + self.epsilon)
