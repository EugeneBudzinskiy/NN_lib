import numpy as np

from nnlibrary.auto_diff_fast import AbstractNode
from .abstractions import ForwardBiOperation
from .abstractions import FrowardUniOperation
from .special_vars import Node


class Addition(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values + x2.values
        partials = x1.partials + x2.partials
        return Node(values=values, partials=partials)


class Subtraction(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values - x2.values
        partials = x1.partials - x2.partials
        return Node(values=values, partials=partials)


class Multiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values * x2.values
        partials = x1.partials * x2.values + x1.values * x2.partials
        return Node(values=values, partials=partials)


class Division(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values / x2.values
        partials = (x1.partials * x2.values - x1.values * x2.partials) / x2.values ** 2
        return Node(values=values, partials=partials)


class Power(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values ** x2.values
        log_x1 = np.log(np.maximum(np.abs(x1.values), Power.epsilon))  # TODO : Probably change partials as well
        partials = x1.values ** (x2.values - 1) * (x1.partials * x2.values + x1.values * x2.partials * log_x1)
        return Node(values=values, partials=partials)


class SquareRoot(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sqrt(x.values)
        partials = x.partials / (2 * values)
        return Node(values=values, partials=partials)


class Exponent(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.exp(x.values)
        partials = x.partials * values
        return Node(values=values, partials=partials)


class Logarithm(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log(x.values)
        partials = x.partials / x.values
        return Node(values=values, partials=partials)


class Logarithm2(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log2(x.values)
        partials = x.partials / (x.values * np.log(2))
        return Node(values=values, partials=partials)


class Logarithm10(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log10(x.values)
        partials = x.partials / (x.values * np.log(10))
        return Node(values=values, partials=partials)


class Sin(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sin(x.values)
        partials = x.partials * np.cos(x.values)
        return Node(values=values, partials=partials)


class Cos(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.cos(x.values)
        partials = - x.partials * np.sin(x.values)
        return Node(values=values, partials=partials)


class Tan(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.tan(x.values)
        partials = x.partials / np.cos(x.values) ** 2
        return Node(values=values, partials=partials)


class Arcsin(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arcsin(x.values)
        partials = x.partials / np.sqrt(1 - x.values ** 2)
        return Node(values=values, partials=partials)


class Arccos(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arccos(x.values)
        partials = - x.partials / np.sqrt(1 - x.values ** 2)
        return Node(values=values, partials=partials)


class Arctan(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arctan(x.values)
        partials = x.partials / (1 + x.values ** 2)
        return Node(values=values, partials=partials)


class Sinh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sinh(x.values)
        partials = x.partials * np.cosh(x.values)
        return Node(values=values, partials=partials)


class Cosh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.cosh(x.values)
        partials = x.partials * np.sinh(x.values)
        return Node(values=values, partials=partials)


class Tanh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.tanh(x.values)
        partials = x.partials * (1 - values ** 2)
        return Node(values=values, partials=partials)


class MatrixMultiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.matmul(x1.values, x2.values, *args, **kwargs)
        partials = np.matmul(x1.partials, x2.values, *args, **kwargs) + \
            np.matmul(x1.values, x2.partials, *args, **kwargs)
        return Node(values=values, partials=partials)


class Summation(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sum(x.values, *args, **kwargs)
        partials = np.sum(x.partials, *args, **kwargs)
        values = values.reshape(1, -1) if values.ndim == 1 else values
        partials = partials.reshape(1, -1) if partials.ndim == 1 else partials
        return Node(values=values, partials=partials)


class Absolute(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.abs(x.values)
        partials = x.partials.copy()
        partials[x.values < 0] *= -1
        return Node(values=values, partials=partials)


class Negative(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = -1 * x.values
        partials = -1 * x.partials
        return Node(values=values, partials=partials)


class Positive(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = +1 * x.values
        partials = +1 * x.partials
        return Node(values=values, partials=partials)
