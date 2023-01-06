from nnlibrary.numpy_wrap import numpy as np

from nnlibrary.numpy_wrap import node
from .abstractions import AbstractOperationUni
from .abstractions import AbstractOperationBi


class Addition(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = x1.values + x2.values
        partials = x1.partials + x2.partials
        return node.Node(values=values, partials=partials)


class Subtraction(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = x1.values - x2.values
        partials = x1.partials - x2.partials
        return node.Node(values=values, partials=partials)


class Multiplication(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = x1.values * x2.values
        partials = x1.partials * x2.values + x1.values * x2.partials
        return node.Node(values=values, partials=partials)


class Division(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = x1.values / x2.values
        partials = (x1.partials * x2.values - x1.values * x2.partials) / x2.values ** 2
        return node.Node(values=values, partials=partials)


class Power(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = x1.values ** x2.values
        log_x1 = np.log(np.maximum(np.abs(x1.values), Power.epsilon))  # TODO : Probably change partials as well
        partials = x1.values ** (x2.values - 1) * (x1.partials * x2.values + x1.values * x2.partials * log_x1)
        return node.Node(values=values, partials=partials)


class MatrixMultiplication(AbstractOperationBi):
    @staticmethod
    def call(x1: node.AbstractNode, x2: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.matmul(x1.values, x2.values, *args, **kwargs)
        partials = np.matmul(x1.partials, x2.values, *args, **kwargs) + \
            np.matmul(x1.values, x2.partials, *args, **kwargs)
        return node.Node(values=values, partials=partials)


class Summation(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.sum(x.values, *args, **kwargs)
        partials = np.sum(x.partials, *args, **kwargs)
        values = values.reshape(1, -1) if values.ndim == 1 else values
        partials = partials.reshape(1, -1) if partials.ndim == 1 else partials
        return node.Node(values=values, partials=partials)


class SquareRoot(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.sqrt(x.values)
        partials = x.partials / (2 * values)
        return node.Node(values=values, partials=partials)


class Exponent(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.exp(x.values)
        partials = x.partials * values
        return node.Node(values=values, partials=partials)


class Logarithm(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.log(x.values)
        partials = x.partials / x.values
        return node.Node(values=values, partials=partials)


class Logarithm2(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.log2(x.values)
        partials = x.partials / (x.values * np.log(2))
        return node.Node(values=values, partials=partials)


class Logarithm10(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.log10(x.values)
        partials = x.partials / (x.values * np.log(10))
        return node.Node(values=values, partials=partials)


class Sin(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.sin(x.values)
        partials = x.partials * np.cos(x.values)
        return node.Node(values=values, partials=partials)


class Cos(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.cos(x.values)
        partials = - x.partials * np.sin(x.values)
        return node.Node(values=values, partials=partials)


class Tan(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.tan(x.values)
        partials = x.partials / np.cos(x.values) ** 2
        return node.Node(values=values, partials=partials)


class Arcsin(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.arcsin(x.values)
        partials = x.partials / np.sqrt(1 - x.values ** 2)
        return node.Node(values=values, partials=partials)


class Arccos(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.arccos(x.values)
        partials = - x.partials / np.sqrt(1 - x.values ** 2)
        return node.Node(values=values, partials=partials)


class Arctan(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.arctan(x.values)
        partials = x.partials / (1 + x.values ** 2)
        return node.Node(values=values, partials=partials)


class Sinh(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.sinh(x.values)
        partials = x.partials * np.cosh(x.values)
        return node.Node(values=values, partials=partials)


class Cosh(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.cosh(x.values)
        partials = x.partials * np.sinh(x.values)
        return node.Node(values=values, partials=partials)


class Tanh(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.tanh(x.values)
        partials = x.partials * (1 - values ** 2)
        return node.Node(values=values, partials=partials)


class Absolute(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = np.abs(x.values)
        partials = x.partials.copy()
        partials[x.values < 0] *= -1
        return node.Node(values=values, partials=partials)


class Negative(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = -1 * x.values
        partials = -1 * x.partials
        return node.Node(values=values, partials=partials)


class Positive(AbstractOperationUni):
    @staticmethod
    def call(x: node.AbstractNode, *args, **kwargs) -> node.AbstractNode:
        values = +1 * x.values
        partials = +1 * x.partials
        return node.Node(values=values, partials=partials)
