from abc import ABC
from abc import abstractmethod

from numpy import ndarray
from numpy import exp
from numpy import tanh


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: ndarray):
        pass


class Linear(Activation):
    def __call__(self, x: ndarray):
        return x.copy()


class Sigmoid(Activation):
    def __call__(self, x: ndarray):
        return 1 / (1 + exp(-x))


class HardSigmoid(Activation):
    def __call__(self, x: ndarray):
        z = x.copy()
        z[(z >= -2.5) * (z <= 2.5)] *= 0.2
        z[(z >= -2.5) * (z <= 2.5)] += 0.5
        z[z < -2.5] = 0
        z[z > 2.5] = 1
        return z


class ReLU(Activation):
    def __call__(self, x: ndarray):
        z = x.copy()
        z[z < 0] = 0
        return z


class TanH(Activation):
    def __call__(self, x: ndarray):
        return tanh(x)


class Exponent(Activation):
    def __call__(self, x: ndarray):
        return exp(x)
