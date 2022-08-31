import numpy as np

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast.forward_mode import FrowardUniOperation
from nnlibrary.auto_diff_fast.forward_mode import ForwardBiOperation
from nnlibrary.auto_diff_fast.forward_mode import special_vars


class Addition(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        values = x1.values + x2.values
        partials = x1.partials + x2.partials
        return special_vars.Node(values=values, partials=partials)


class Multiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        values = x1.values * x2.values
        partials = x1.partials * x2.values + x1.values * x2.partials
        return special_vars.Node(values=values, partials=partials)


class MatrixMultiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        values = np.dot(x1.values, x2.values)
        partials = np.dot(x1.partials, x2.values) + np.dot(x1.values, x2.partials)
        return special_vars.Node(values=values, partials=partials)


class Summation(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        values = np.array([np.sum(x.values)])
        partials = x.partials
        return special_vars.Node(values=values, partials=partials)
