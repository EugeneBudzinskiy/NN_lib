from abc import ABC
from abc import abstractmethod

from SingletonMeta import SingletonMeta


class Activations(metaclass=SingletonMeta):
    def __init__(self):
        self.ReLU = ReLU()


class AbstractActivation(ABC):
    @abstractmethod
    def activation(self):
        pass

    @abstractmethod
    def derivative(self):
        pass


class ReLU(AbstractActivation):
    def __init__(self):
        pass

    def activation(self):
        pass

    def derivative(self):
        pass
