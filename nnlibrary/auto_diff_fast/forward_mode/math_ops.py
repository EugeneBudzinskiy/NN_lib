import numpy as np

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast.forward_mode import FrowardUniOperation
from nnlibrary.auto_diff_fast.forward_mode import ForwardBiOperation
from nnlibrary.auto_diff_fast.forward_mode import special_vars


class Addition(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values + x2.values
        partials = x1.partials + x2.partials
        return special_vars.Node(values=values, partials=partials)


class Multiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values * x2.values
        partials = x1.partials * x2.values + x1.values * x2.partials
        return special_vars.Node(values=values, partials=partials)


class MatrixMultiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.dot(x1.values, x2.values)
        partials = np.dot(x1.partials, x2.values) + np.dot(x1.values, x2.partials)
        return special_vars.Node(values=values, partials=partials)


class Summation(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sum(x.values, *args, **kwargs)
        partials = np.sum(x.partials, *args, **kwargs)
        values = values.reshape(1, -1) if values.ndim == 1 else values
        partials = partials.reshape(1, -1) if partials.ndim == 1 else partials
        return special_vars.Node(values=values, partials=partials)
