import numpy as np

from .abstractions import AbstractNode
from .abstractions import AbstractMathOperationUni
from .abstractions import AbstractMathOperationBi
from .node import Node


class Addition(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values + x2.values
        partials = x1.partials + x2.partials
        return Node(values=values, partials=partials)


class Subtraction(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values - x2.values
        partials = x1.partials - x2.partials
        return Node(values=values, partials=partials)


class Multiplication(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values * x2.values
        partials = x1.partials * x2.values + x1.values * x2.partials
        return Node(values=values, partials=partials)


class Division(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values / x2.values
        partials = (x1.partials * x2.values - x1.values * x2.partials) / x2.values ** 2
        return Node(values=values, partials=partials)


class Power(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = x1.values ** x2.values
        log_x1 = np.log(np.maximum(np.abs(x1.values), Power.epsilon))  # TODO : Probably change partials as well
        partials = x1.values ** (x2.values - 1) * (x1.partials * x2.values + x1.values * x2.partials * log_x1)
        return Node(values=values, partials=partials)


class SquareRoot(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sqrt(x.values)
        partials = x.partials / (2 * values)
        return Node(values=values, partials=partials)


class Exponent(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.exp(x.values)
        partials = x.partials * values
        return Node(values=values, partials=partials)


class Logarithm(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log(x.values)
        partials = x.partials / x.values
        return Node(values=values, partials=partials)


class Logarithm2(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log2(x.values)
        partials = x.partials / (x.values * np.log(2))
        return Node(values=values, partials=partials)


class Logarithm10(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.log10(x.values)
        partials = x.partials / (x.values * np.log(10))
        return Node(values=values, partials=partials)


class Sin(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sin(x.values)
        partials = x.partials * np.cos(x.values)
        return Node(values=values, partials=partials)


class Cos(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.cos(x.values)
        partials = - x.partials * np.sin(x.values)
        return Node(values=values, partials=partials)


class Tan(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.tan(x.values)
        partials = x.partials / np.cos(x.values) ** 2
        return Node(values=values, partials=partials)


class Arcsin(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arcsin(x.values)
        partials = x.partials / np.sqrt(1 - x.values ** 2)
        return Node(values=values, partials=partials)


class Arccos(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arccos(x.values)
        partials = - x.partials / np.sqrt(1 - x.values ** 2)
        return Node(values=values, partials=partials)


class Arctan(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.arctan(x.values)
        partials = x.partials / (1 + x.values ** 2)
        return Node(values=values, partials=partials)


class Sinh(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sinh(x.values)
        partials = x.partials * np.cosh(x.values)
        return Node(values=values, partials=partials)


class Cosh(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.cosh(x.values)
        partials = x.partials * np.sinh(x.values)
        return Node(values=values, partials=partials)


class Tanh(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.tanh(x.values)
        partials = x.partials * (1 - values ** 2)
        return Node(values=values, partials=partials)


class MatrixMultiplication(AbstractMathOperationBi):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.matmul(x1.values, x2.values, *args, **kwargs)
        partials = np.matmul(x1.partials, x2.values, *args, **kwargs) + \
            np.matmul(x1.values, x2.partials, *args, **kwargs)
        return Node(values=values, partials=partials)


class Summation(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.sum(x.values, *args, **kwargs)
        partials = np.sum(x.partials, *args, **kwargs)
        values = values.reshape(1, -1) if values.ndim == 1 else values
        partials = partials.reshape(1, -1) if partials.ndim == 1 else partials
        return Node(values=values, partials=partials)


class Absolute(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = np.abs(x.values)
        partials = x.partials.copy()
        partials[x.values < 0] *= -1
        return Node(values=values, partials=partials)


class Negative(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = -1 * x.values
        partials = -1 * x.partials
        return Node(values=values, partials=partials)


class Positive(AbstractMathOperationUni):
    @staticmethod
    def call(x: AbstractNode, *args, **kwargs) -> AbstractNode:
        values = +1 * x.values
        partials = +1 * x.partials
        return Node(values=values, partials=partials)
