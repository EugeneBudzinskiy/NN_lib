from abc import ABC
from abc import abstractmethod

from nnlibrary.auto_diff.variables import AbstractVariable


class AbstractOperation(ABC):
    def __init__(self):
        self.epsilon = 1e-7

    @abstractmethod
    def __call__(self, *args, **kwargs) -> AbstractVariable:
        pass


class UniOperation(AbstractOperation):
    @abstractmethod
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        pass


class BiOperation(AbstractOperation):
    @abstractmethod
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        pass
