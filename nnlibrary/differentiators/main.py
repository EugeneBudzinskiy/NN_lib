import numpy as np

from nnlibrary.differentiators import AbstractDifferentiator


class Derivative(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


class Gradient(AbstractDifferentiator):
    def __call__(self, func: callable, x: np.ndarray,  epsilon: float = 1e-5) -> np.ndarray:
        point = x.copy().reshape((1, -1)) if x.ndim == 1 else x.copy()
        tmp = np.zeros_like(point, dtype='float64')
        output = tmp.copy()
        for i in range(x.shape[-1]):
            offset = tmp.copy()
            offset[:, i] = epsilon
            output[:, i] = (func(point + offset) - func(x - offset)) / (2 * epsilon)

        return output
