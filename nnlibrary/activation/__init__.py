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
        return x

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
        x[(x >= -2.5) * (x <= 2.5)] *= 0.2
        x[(x >= -2.5) * (x <= 2.5)] += 0.5
        x[x < -2.5] = 0
        x[x > 2.5] = 1
        return x

    @staticmethod
    def derivative(x):
        e = Sigmoid.activate(x)
        return e * (1 - e)


class ReLU(AbstractActivation):
    @staticmethod
    def activate(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


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
