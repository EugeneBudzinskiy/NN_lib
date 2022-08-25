from abc import ABC
from abc import abstractmethod


class AbstractVariable(ABC):
    def __init__(self, value: float, gradient: float = 0.):
        self.value = value
        self.gradient = gradient

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __pos__(self):
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
