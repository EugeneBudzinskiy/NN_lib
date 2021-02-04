import numpy as np
from abc import ABC
from abc import abstractmethod

from nnlibrary.singleton import SingletonMeta


class Activations(metaclass=SingletonMeta):
    def __init__(self):
        self.Linear = Linear()
        self.Sigmoid = Sigmoid()
        self.HardSigmoid = HardSigmoid()
        self.ReLU = ReLU()
        self.TanH = TanH()
        self.Exponential = Exponential()


class AbstractActivation(ABC, metaclass=SingletonMeta):
    @staticmethod
    @abstractmethod
    def activate(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass


class Linear(AbstractActivation):
    @staticmethod
    def activate(x):
        return x.copy()

    @staticmethod
    def derivative(x):
        return np.ones_like(x)


class Sigmoid(AbstractActivation):
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        e = Sigmoid.activate(x)
        return e * (1 - e)


class HardSigmoid(AbstractActivation):
    @staticmethod
    def activate(x):
        z = x.copy()
        z[(z >= -2.5) * (z <= 2.5)] *= 0.2
        z[(z >= -2.5) * (z <= 2.5)] += 0.5
        z[z < -2.5] = 0
        z[z > 2.5] = 1
        return z

    @staticmethod
    def derivative(x):
        z = x.copy()
        z[(z >= -2.5) * (z <= 2.5)] = 0.2
        z[z > 2.5] = 0
        z[z < -2.5] = 0
        return z


class ReLU(AbstractActivation):
    @staticmethod
    def activate(x):
        z = x.copy()
        z[z < 0] = 0
        return z

    @staticmethod
    def derivative(x):
        z = x.copy()
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class TanH(AbstractActivation):
    @staticmethod
    def activate(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.square(TanH.activate(x))


class Exponential(AbstractActivation):
    @staticmethod
    def activate(x):
        return np.exp(x)

    @staticmethod
    def derivative(x):
        return Exponential.activate(x)
