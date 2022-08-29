from abc import abstractmethod

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import AbstractSpecialOperation


class AbstractNodeOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(*args, **kwargs) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractSpecialVariable:
        pass


class ReverseUniOperation(AbstractNodeOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass


class ReverseBiOperation(AbstractNodeOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        pass
