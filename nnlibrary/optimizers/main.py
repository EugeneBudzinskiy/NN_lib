import numpy as np

from nnlibrary.optimizers import AbstractOptimizer


class SGD(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 momentum: float = 0.0,
                 nesterov: bool = False):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = 0

    def __call__(self, gradient_vector: np.ndarray):
        if self.momentum == 0:
            adjustment = - self.learning_rate * gradient_vector
        else:
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient_vector
            adjustment = self.velocity

            if self.nesterov:
                adjustment = self.momentum * self.velocity - self.learning_rate * gradient_vector

        return adjustment.copy()


class Adam(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.v_t = 0
        self.s_t = 0

        self.powered_beta_1 = 1
        self.powered_beta_2 = 1

    def __call__(self, gradient_vector: np.ndarray):
        self.v_t = self.beta_1 * self.v_t + (1 - self.beta_1) * gradient_vector
        self.s_t = self.beta_2 * self.s_t + (1 - self.beta_2) * gradient_vector ** 2

        self.powered_beta_1 *= self.beta_1
        self.powered_beta_2 *= self.beta_2

        dash_v_t = self.v_t / (1 - self.powered_beta_1)
        dash_s_t = self.s_t / (1 - self.powered_beta_2)

        return - self.learning_rate * dash_v_t / (np.sqrt(dash_s_t) + self.epsilon)


class RMSprop(AbstractOptimizer):
    def __init__(self,
                 learning_rate: float = 0.0001,
                 rho: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-7):

        self.learning_rate = learning_rate
        self.beta = rho
        self.momentum = momentum
        self.epsilon = epsilon

        self.previous = 0
        self.velocity = 0

    def __call__(self, gradient_vector: np.ndarray) -> np.ndarray:
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient_vector ** 2

        direction = - self.learning_rate * gradient_vector / (np.sqrt(self.velocity + self.epsilon))
        adjustment = direction

        if self.momentum:
            self.previous = self.momentum * self.previous + direction
            adjustment = self.previous

        return adjustment.copy()
