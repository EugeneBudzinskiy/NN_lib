from abc import abstractmethod

from nnlibrary.auto_diff import AbstractSpecialOperation
from nnlibrary.auto_diff import AbstractSpecialVariable


class FrowardUniOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass


class ForwardBiOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass
