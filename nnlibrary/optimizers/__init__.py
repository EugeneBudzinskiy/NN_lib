from abc import ABC
from abc import abstractmethod

from nnlibrary.singleton import SingletonMeta


class Optimizers(metaclass=SingletonMeta):
    def __init__(self):
        self.SGD = SGD


class Optimizer(ABC):
    @abstractmethod
    def optimize(self, training_variables, gradient_vector):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.0001):
        self.learning_rate = learning_rate

    def optimize(self, training_variables, gradient_vector):
        training_variables += self.learning_rate * gradient_vector
