from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractNode(ABC):
    __slots__ = ['values', 'partials']

    def __init__(self, values: np.ndarray, partials: np.ndarray = None):
        self.values = values
        self.partials = np.zeros_like(values) if partials is None else  partials

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


class AbstractSpecialOperation(ABC):
    epsilon = 1e-7

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractNode:
        pass


class AbstractMode(ABC):
    pass
