from abc import abstractmethod

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast import AbstractSpecialOperation


class AbstractNodeOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(*args, **kwargs) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(*args, **kwargs) -> AbstractNode:
        pass


class ReverseUniOperation(AbstractNodeOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(x: AbstractNode) -> AbstractNode:
        pass


class ReverseBiOperation(AbstractNodeOperation):
    @staticmethod
    @abstractmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        pass
