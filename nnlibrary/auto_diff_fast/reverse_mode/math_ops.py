import numpy as np

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast.reverse_mode import ReverseUniOperation
from nnlibrary.auto_diff_fast.reverse_mode import ReverseBiOperation
from nnlibrary.auto_diff_fast.reverse_mode import special_vars


class Addition(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        return 1, 1

    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value + x2.value
        inputs_partials = Addition.get_inputs_partials(x1=x1, x2=x2)
        return special_vars.Operator(value=value, inputs=(x1, x2), inputs_partials=inputs_partials)


class Subtraction(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        return 1, -1

    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value - x2.value
        inputs_partials = Subtraction.get_inputs_partials(x1=x1, x2=x2)
        return special_vars.Operator(value=value, inputs=(x1, x2), inputs_partials=inputs_partials)


class Multiplication(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        return x2.value, x1.value

    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value * x2.value
        inputs_partials = Multiplication.get_inputs_partials(x1=x1, x2=x2)
        return special_vars.Operator(value=value, inputs=(x1, x2), inputs_partials=inputs_partials)


class Division(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        return 1. / x2.value, - x1.value / x2.value ** 2

    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value / x2.value
        inputs_partials = Division.get_inputs_partials(x1=x1, x2=x2)
        return special_vars.Operator(value=value, inputs=(x1, x2), inputs_partials=inputs_partials)


class Power(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractNode, x2: AbstractNode) -> tuple:
        log_x1 = np.log(Power.epsilon) if abs(x1.value) < Power.epsilon else np.log(abs(x1.value))
        return x1.value ** (x2.value - 1) * x2.value, x1.value ** x2.value * log_x1

    @staticmethod
    def call(x1: AbstractNode, x2: AbstractNode) -> AbstractNode:
        value = x1.value ** x2.value
        inputs_partials = Power.get_inputs_partials(x1=x1, x2=x2)
        return special_vars.Operator(value=value, inputs=(x1, x2), inputs_partials=inputs_partials)


class Absolute(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return -1 if x.value < 0 else 1,

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.abs(x.value)
        inputs_partials = Negative.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Negative(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return -1,

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = -1 * x.value
        inputs_partials = Negative.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Positive(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1,

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = +1 * x.value
        inputs_partials = Positive.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class SquareRoot(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / (2 * np.sqrt(x.value)),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sqrt(x.value)
        inputs_partials = SquareRoot.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Exponent(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return np.exp(x.value),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.exp(x.value)
        inputs_partials = Exponent.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Logarithm(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / x.value,

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log(x.value)
        inputs_partials = Logarithm.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Logarithm2(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / (x.value * np.log(2)),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log2(x.value)
        inputs_partials = Logarithm2.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Logarithm10(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / (x.value * np.log(10)),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.log10(x.value)
        inputs_partials = Logarithm10.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Sin(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return np.cos(x.value),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sin(x.value)
        inputs_partials = Sin.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Cos(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return - np.sin(x.value),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.cos(x.value)
        inputs_partials = Cos.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Tan(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / np.cos(x.value) ** 2,

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.tan(x.value)
        inputs_partials = Tan.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Arcsin(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / np.sqrt(1 - x.value ** 2),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arcsin(x.value)
        inputs_partials = Arcsin.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Arccos(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return -1 / np.sqrt(1 - x.value ** 2),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arccos(x.value)
        inputs_partials = Arccos.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Arctan(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return 1 / (1 + x.value ** 2),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.arctan(x.value)
        inputs_partials = Arctan.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Sinh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return np.cosh(x.value),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.sinh(x.value)
        inputs_partials = Sinh.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Cosh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return np.sinh(x.value),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.cosh(x.value)
        inputs_partials = Cosh.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)


class Tanh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractNode) -> tuple:
        return (1 - np.tanh(x.value) ** 2),

    @staticmethod
    def call(x: AbstractNode) -> AbstractNode:
        value = np.tanh(x.value)
        inputs_partials = Tanh.get_inputs_partials(x=x)
        return special_vars.Operator(value=value, inputs=(x, ), inputs_partials=inputs_partials)

