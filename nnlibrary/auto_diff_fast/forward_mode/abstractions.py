from abc import abstractmethod

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast import AbstractSpecialOperation


class FrowardUniOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def call(x: AbstractNode) -> AbstractNode:
        pass


class ForwardBiOperation(AbstractSpecialOperation):
    @staticmethod
    @abstractmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        pass
