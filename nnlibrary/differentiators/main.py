import numpy as np

from nnlibrary.differentiators import AbstractDifferentiator

# TODO SimpleDiff == Derivative, Diff == Gradient


class SimpleDifferentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


class Differentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        epsilon_matrix = np.diag(np.ones_like(x.flatten())) * epsilon
        return (func(x + epsilon_matrix) - func(x - epsilon_matrix)) / (2 * epsilon)
