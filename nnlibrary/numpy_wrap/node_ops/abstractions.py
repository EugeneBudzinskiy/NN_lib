from abc import ABC
from abc import abstractmethod

from nnlibrary import numpy_wrap as npw
from nnlibrary.numpy_wrap.node import AbstractNode


class AbstractOperation(ABC):
    epsilon = 1e-7

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractNode:
        pass


class AbstractOperationUni(AbstractOperation):
    @staticmethod
    @abstractmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        pass


class AbstractOperationBi(AbstractOperation):
    @staticmethod
    @abstractmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        pass
