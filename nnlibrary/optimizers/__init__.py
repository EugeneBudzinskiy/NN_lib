from numpy import ndarray
from numpy import sqrt
from abc import ABC
from abc import abstractmethod

from nnlibrary.singleton import SingletonMeta


class Optimizers(metaclass=SingletonMeta):
    def __init__(self):
        self.SGD = SGD
        self.Adam = Adam


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, trainable_variables: ndarray, gradient_vector: ndarray):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.0001):
        self.learning_rate = learning_rate

    def optimize(self, trainable_variables: ndarray, gradient_vector: ndarray):
        trainable_variables -= self.learning_rate * gradient_vector


class Adam(Optimizer):
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

    def optimize(self, trainable_variables: ndarray, gradient_vector: ndarray):
        self.v_t = self.beta_1 * self.v_t + (1 - self.beta_1) * gradient_vector
        self.s_t = self.beta_2 * self.s_t + (1 - self.beta_2) * gradient_vector ** 2

        self.powered_beta_1 *= self.beta_1
        self.powered_beta_2 *= self.beta_2

        dash_v_t = self.v_t / (1 - self.powered_beta_1)
        dash_s_t = self.s_t / (1 - self.powered_beta_2)

        trainable_variables -= self.learning_rate * dash_v_t / (sqrt(dash_s_t) + self.epsilon)

