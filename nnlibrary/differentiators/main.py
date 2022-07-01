from numpy import ndarray

from nnlibrary.differentiators.abstractions import AbstractDifferentiator


class SimpleDifferentiator(AbstractDifferentiator):
    def __call__(self, func: callable, x: ndarray,  epsilon: float = 1e-5):
        return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)  # Rewrite to multi-dim
