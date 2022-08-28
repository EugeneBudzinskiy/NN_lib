from abc import abstractmethod

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import AbstractSpecialOperation


class ReverseUniOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        pass

    @staticmethod
    @abstractmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass


class ReverseBiOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        pass

    @staticmethod
    @abstractmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass
