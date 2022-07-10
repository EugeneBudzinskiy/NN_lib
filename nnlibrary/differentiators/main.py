import numpy as np

from nnlibrary.differentiators.abstractions import AbstractDifferentiator


class SimpleDifferentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


class Differentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        epsilon_matrix = np.diag(np.ones_like(x)) * epsilon
        point = x.copy().reshape((-1, 1))
        return (func(point + epsilon_matrix) - func(point - epsilon_matrix)) / (2 * epsilon)
