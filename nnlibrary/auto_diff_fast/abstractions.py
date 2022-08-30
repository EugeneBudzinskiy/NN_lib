from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractNode(ABC):
    __slots__ = []

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _init_root(self, *args, **kwargs):
        pass

    @classmethod
    def create_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root._init_root(*args, **kwargs)
        return root

    @staticmethod
    @abstractmethod
    def _wrapper(other):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __radd__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __rsub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __rmul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        pass

    @abstractmethod
    def __pow__(self, power, modulo=None):
        pass

    @abstractmethod
    def __rpow__(self, power, modulo=None):
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
    def __abs__(self):
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


class AbstractSpecialOperation(ABC):
    epsilon = 1e-7

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractNode:
        pass


class AbstractMode(ABC):
    @staticmethod
    def partial_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.partial)(x)

    @staticmethod
    def value_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.value)(x)

    @staticmethod
    def set_partial(var_x: np.ndarray, value: float):
        for i in range(var_x.shape[-1]):
            var_x[i].partial = value

    @staticmethod
    @abstractmethod
    def to_variable(x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def to_variable_direction(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        pass
