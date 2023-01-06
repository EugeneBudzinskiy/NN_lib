import numpy as np

from .abstractions import AbstractActivation


class Linear(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.copy()


class Sigmoid(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class HardSigmoid(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[(z >= -2.5) * (z <= 2.5)] *= 0.2
        z[(z >= -2.5) * (z <= 2.5)] += 0.5
        z[z < -2.5] = 0
        z[z > 2.5] = 1
        return z


class ReLU(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        z[z < 0] = 0
        return z


class TanH(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


class Exponent(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


class Softmax(AbstractActivation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1).reshape(-1, 1)
