import numpy as np

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast.forward_mode import FrowardUniOperation
from nnlibrary.auto_diff_fast.forward_mode import ForwardBiOperation
from nnlibrary.auto_diff_fast.forward_mode import special_vars


class Addition(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value + x2.value
        partial = x1.partial + x2.partial
        return special_vars.Node(value=value, partial=partial)


class Subtraction(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value - x2.value
        partial = x1.partial - x2.partial
        return special_vars.Node(value=value, partial=partial)


class Absolute(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.abs(x.value)
        partial = (-1 * x.partial) if x.value < 0 else (+1 * x.partial)
        return special_vars.Node(value=value, partial=partial)


class Negative(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = -1 * x.value
        partial = -1 * x.partial
        return special_vars.Node(value=value, partial=partial)


class Positive(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = +1 * x.value
        partial = +1 * x.partial
        return special_vars.Node(value=value, partial=partial)


class Multiplication(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value * x2.value
        partial = x1.partial * x2.value + x1.value * x2.partial
        return special_vars.Node(value=value, partial=partial)


class Division(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value / x2.value
        partial = (x1.partial * x2.value - x1.value * x2.partial) / x2.value ** 2
        return special_vars.Node(value=value, partial=partial)


class Power(ForwardBiOperation):
    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value ** x2.value
        log_x1 = np.log(Power.epsilon) if abs(x1.value) < Power.epsilon else np.log(abs(x1.value))
        partial = x1.value ** (x2.value - 1) * (x1.partial * x2.value + x1.value * x2.partial * log_x1)
        return special_vars.Node(value=value, partial=partial)


class SquareRoot(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sqrt(x.value)
        partial = x.partial / (2 * value)
        return special_vars.Node(value=value, partial=partial)


class Exponent(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.exp(x.value)
        partial = x.partial * value
        return special_vars.Node(value=value, partial=partial)


class Logarithm(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log(x.value)
        partial = x.partial / x.value
        return special_vars.Node(value=value, partial=partial)


class Logarithm2(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log2(x.value)
        partial = x.partial / (x.value * np.log(2))
        return special_vars.Node(value=value, partial=partial)


class Logarithm10(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log10(x.value)
        partial = x.partial / (x.value * np.log(10))
        return special_vars.Node(value=value, partial=partial)


class Sin(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sin(x.value)
        partial = x.partial * np.cos(x.value)
        return special_vars.Node(value=value, partial=partial)


class Cos(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.cos(x.value)
        partial = - x.partial * np.sin(x.value)
        return special_vars.Node(value=value, partial=partial)


class Tan(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.tan(x.value)
        partial = x.partial / np.cos(x.value) ** 2
        return special_vars.Node(value=value, partial=partial)


class Arcsin(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arcsin(x.value)
        partial = x.partial / np.sqrt(1 - x.value ** 2)
        return special_vars.Node(value=value, partial=partial)


class Arccos(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arccos(x.value)
        partial = - x.partial / np.sqrt(1 - x.value ** 2)
        return special_vars.Node(value=value, partial=partial)


class Arctan(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arctan(x.value)
        partial = x.partial / (1 + x.value ** 2)
        return special_vars.Node(value=value, partial=partial)


class Sinh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sinh(x.value)
        partial = x.partial * np.cosh(x.value)
        return special_vars.Node(value=value, partial=partial)


class Cosh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.cosh(x.value)
        partial = x.partial * np.sinh(x.value)
        return special_vars.Node(value=value, partial=partial)


class Tanh(FrowardUniOperation):
    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.tanh(x.value)
        partial = x.partial * (1 - value ** 2)
        return special_vars.Node(value=value, partial=partial)
