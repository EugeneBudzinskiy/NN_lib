import numpy as np
from abc import ABC
from abc import abstractmethod

from singleton import SingletonMeta


class Activations(metaclass=SingletonMeta):
    def __init__(self):
        self.Sigmoid = Sigmoid()
        self.ReLU = ReLU()
        self.TanH = TanH()
        self.Linear = Linear()


class AbstractActivation(ABC, metaclass=SingletonMeta):
    @staticmethod
    @abstractmethod
    def activation(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass


class Sigmoid(AbstractActivation):
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        e = Sigmoid.activation(x)
        return e * (1 - e)


class ReLU(AbstractActivation):
    @staticmethod
    def activation(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def derivative(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x


class TanH(AbstractActivation):
    @staticmethod
    def activation(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.square(TanH.activation(x))


class Linear(AbstractActivation):
    @staticmethod
    def activation(x):
        return x

    @staticmethod
    def derivative(x):
        return 1
