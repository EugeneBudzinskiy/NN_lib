from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class AbstractDifferentiator(ABC):
    @abstractmethod
    def __call__(self, func: callable, x: ndarray, epsilon: float = 1e-5):
        pass
