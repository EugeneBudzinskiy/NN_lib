import numpy as np

from typing import Callable

from .abstractions import AbstractDifferentiator


class Derivative(AbstractDifferentiator):
    def __call__(self,
                 func: Callable[[np.ndarray], np.ndarray],
                 x: np.ndarray,
                 epsilon: float = 1e-5) -> np.ndarray:
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


class Gradient(AbstractDifferentiator):
    def __call__(self,
                 func: Callable[[np.ndarray], np.ndarray],
                 x: np.ndarray,
                 epsilon: float = 1e-5) -> np.ndarray:
        point = x.copy() if x.ndim > 1 else x.copy().reshape(1, -1)
        tmp = np.zeros_like(point, dtype='float64')
        output = tmp.copy()
        for i in range(point.shape[-1]):
            offset = tmp.copy()
            offset[:, i] = epsilon
            output[:, i] = (func(point + offset) - func(point - offset)) / (2 * epsilon)

        return output
