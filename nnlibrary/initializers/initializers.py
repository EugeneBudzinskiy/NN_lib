import numpy as np

from .abstractions import AbstractInitializer


class Zeros(AbstractInitializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape=shape)


class UniformZeroOne(AbstractInitializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.random.random(size=shape)


class Uniform(AbstractInitializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return 2 * np.random.random(size=shape) - 1
