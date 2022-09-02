from abc import ABC
from abc import abstractmethod

from nnlibrary import numpy_wrap as npw


class AbstractNode(ABC):
    __slots__ = ['values', 'partials']

    def __init__(self, values: npw.typing.NDArray, partials: npw.typing.NDArray):
        self.values = values
        self.partials = partials

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

