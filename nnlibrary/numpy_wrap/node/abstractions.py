from abc import ABC
from abc import abstractmethod

from nnlibrary import numpy_wrap as npw


class AbstractNode(ABC):
    __slots__ = ['values', 'partials']

    def __init__(self, values: npw.ndarray, partials: npw.ndarray = None):
        self.values = values
        self.partials = npw.numpy.zeros_like(values) if partials is None else partials

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
