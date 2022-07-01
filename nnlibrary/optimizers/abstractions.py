from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class AbstractOptimizer(ABC):
    @abstractmethod
    def __call__(self, trainable_variables: ndarray, gradient_vector: ndarray):
        pass
