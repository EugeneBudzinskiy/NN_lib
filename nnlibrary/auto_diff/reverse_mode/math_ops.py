import numpy as np

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import UniOperation
from nnlibrary.auto_diff import BiOperation

from nnlibrary.auto_diff.reverse_mode import special_vars


class Addition(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial + x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value + x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Subtraction(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial - x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value - x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Negative(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return -1 * x.partial

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = -1 * x.value
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Positive(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return +1 * x.partial

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = +1 * x.value
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Multiplication(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial * x2.value + x1.value * x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value * x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Division(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return (x1.partial * x2.value - x1.value * x2.partial) / x2.value ** 2

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value / x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Power(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        log_x1 = np.log(Power.epsilon) if abs(x1.value) < Power.epsilon else np.log(x1.value)
        return x1.value ** (x2.value - 1) * (x1.partial * x2.value + x1.value * x2.partial * log_x1)

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value ** x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class SquareRoot(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (2 * np.sqrt(x.value))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sqrt(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Exponent(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.exp(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.exp(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / x.value

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm2(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (x.value * np.log(2))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log2(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm10(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (x.value * np.log(10))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log10(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Sin(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.cos(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sin(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Cos(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return - x.partial * np.sin(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cos(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Tan(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / np.cos(x.value) ** 2

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tan(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arcsin(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arcsin(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arccos(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return - x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arccos(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arctan(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (1 + x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arctan(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Sinh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.cosh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sinh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Cosh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.sinh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cosh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Tanh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * (1 - np.tanh(x.value) ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tanh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))

