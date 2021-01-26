from abc import ABC
from abc import abstractmethod

from SingletonMeta import SingletonMeta


class Optimizers(metaclass=SingletonMeta):
    def __init__(self):
        self.SGD = SGD()


class AbstractOptimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass


class SGD(AbstractOptimizer):
    def __init__(self):
        pass

    def optimize(self):
        pass
