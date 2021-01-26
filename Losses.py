from abc import ABC
from abc import abstractmethod

from SingletonMeta import SingletonMeta


class Losses(metaclass=SingletonMeta):
    def __init__(self):
        self.MSE = MSE()


class AbstractLoss(ABC):
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def derivative(self):
        pass


class MSE(AbstractLoss):
    def __init__(self):
        pass

    def loss(self):
        pass

    def derivative(self):
        pass

