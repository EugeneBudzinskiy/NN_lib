from abc import ABC
from abc import abstractmethod

from nnlibrary import numpy_wrap as npw


class AbstractNode(ABC):
    __slots__ = ['values', 'partials']

    def __init__(self, values: npw.ndarray, partials: npw.ndarray = None):
        self.values = values
        self.partials = npw.numpy.zeros_like(values) if partials is None else partials

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
    def __mul__(self, other):
        pass

    @abstractmethod
    def __rmul__(self, other):
        pass


class AbstractMathOperation(ABC):
    epsilon = 1e-7

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractNode:
        pass


class AbstractMathOperationUni(AbstractMathOperation):
    @staticmethod
    @abstractmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        pass


class AbstractMathOperationBi(AbstractMathOperation):
    @staticmethod
    @abstractmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        pass
