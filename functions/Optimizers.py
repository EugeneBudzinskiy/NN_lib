from abc import ABC
from abc import abstractmethod

from singleton import SingletonMeta


class Optimizers(metaclass=SingletonMeta):
    def __init__(self):
        self.SGD = SGD


class AbstractOptimizer(ABC):
    @abstractmethod
    def optimize(self, training_variables, gradient_vector):
        pass


class SGD(AbstractOptimizer):
    def __init__(self, learning_rate: int = 0.0001):
        self.learning_rate = learning_rate

    def optimize(self, training_variables, gradient_vector):
        training_variables += self.learning_rate * gradient_vector
