from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class Differentiator(ABC):
    @abstractmethod
    def __call__(self, func: callable, x: ndarray, epsilon: float = 1e-5):
        pass


class SimpleDifferentiator(Differentiator):
    def __call__(self, func: callable, x: ndarray,  epsilon: float = 1e-5):
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)  # Rewrite to multi-dim
