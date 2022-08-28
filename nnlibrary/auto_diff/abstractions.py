from abc import ABC
from abc import abstractmethod

from typing import Any


class AbstractSpecialVariable(ABC):
    def __init__(self, value: float, partial: float = 0.):
        self.value = value
        self.partial = partial

    @abstractmethod
    def _wrapper(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def __pow__(self, power, modulo=None):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __ne__(self, other):
        pass

    @abstractmethod
    def __le__(self, other):
        pass

    @abstractmethod
    def __ge__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __gt__(self, other):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __pos__(self):
        pass

    @abstractmethod
    def sqrt(self):
        pass

    @abstractmethod
    def exp(self):
        pass

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def log2(self):
        pass

    @abstractmethod
    def log10(self):
        pass

    @abstractmethod
    def sin(self):
        pass

    @abstractmethod
    def cos(self):
        pass

    @abstractmethod
    def tan(self):
        pass

    @abstractmethod
    def arcsin(self):
        pass

    @abstractmethod
    def arccos(self):
        pass

    @abstractmethod
    def arctan(self):
        pass

    @abstractmethod
    def sinh(self):
        pass

    @abstractmethod
    def cosh(self):
        pass

    @abstractmethod
    def tanh(self):
        pass


class AbstractOperation(ABC):
    epsilon = 1e-7

    @staticmethod
    @abstractmethod
    def partial(*args, **kwargs) -> float:
        pass

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractSpecialVariable:
        pass


class UniOperation(AbstractOperation):
    @staticmethod
    @abstractmethod
    def partial(x: AbstractSpecialVariable) -> float:
        pass

    @staticmethod
    @abstractmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass


class BiOperation(AbstractOperation):
    @staticmethod
    @abstractmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        pass

    @staticmethod
    @abstractmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass
