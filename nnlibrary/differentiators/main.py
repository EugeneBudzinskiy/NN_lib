import numpy as np

from nnlibrary.differentiators import AbstractDifferentiator


class SimpleDifferentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


class Differentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        result = np.zeros_like(x)
        epsilon_matrix = np.diag(np.ones_like(x[0])) * epsilon
        for i in range(x.shape[0]):
            point = x[i].reshape((1, -1))

            buff = (func(point + epsilon_matrix) - func(point - epsilon_matrix)) / (2 * epsilon)
            result[i] = np.diag(buff).reshape(1, -1) if buff.shape[0] == buff.shape[1] else buff
        return result
