from numpy import ndarray
from numpy import exp
from numpy import tanh

from nnlibrary.activations.abstractions import AbstractActivation


class Linear(AbstractActivation):
    def __call__(self, x: ndarray):
        return x.copy()


class Sigmoid(AbstractActivation):
    def __call__(self, x: ndarray):
        return 1 / (1 + exp(-x))


class HardSigmoid(AbstractActivation):
    def __call__(self, x: ndarray):
        z = x.copy()
        z[(z >= -2.5) * (z <= 2.5)] *= 0.2
        z[(z >= -2.5) * (z <= 2.5)] += 0.5
        z[z < -2.5] = 0
        z[z > 2.5] = 1
        return z


class ReLU(AbstractActivation):
    def __call__(self, x: ndarray):
        z = x.copy()
        z[z < 0] = 0
        return z


class TanH(AbstractActivation):
    def __call__(self, x: ndarray):
        return tanh(x)


class Exponent(AbstractActivation):
    def __call__(self, x: ndarray):
        return exp(x)
