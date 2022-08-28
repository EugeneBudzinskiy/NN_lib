import numpy as np

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff.reverse_mode import ReverseUniOperation
from nnlibrary.auto_diff.reverse_mode import ReverseBiOperation
from nnlibrary.auto_diff.reverse_mode import special_vars


class Addition(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        return 1, 1

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value + x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Subtraction(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        return 1, -1

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value - x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Negative(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return tuple([-1.])

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = -1 * x.value
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Positive(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return tuple([1.])

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = +1 * x.value
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Multiplication(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        return x2.value, x1.value

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value * x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Division(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        return 1. / x2.value, - x1.value / x2.value ** 2

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value / x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class Power(ReverseBiOperation):
    @staticmethod
    def get_inputs_partials(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> tuple[float, float]:
        log_x1 = np.log(Power.epsilon) if abs(x1.value) < Power.epsilon else np.log(x1.value)
        return x1.value ** (x2.value - 1) * x2.value, x1.value ** x2.value * log_x1

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value ** x2.value
        return special_vars.Operator(value=value, inputs=(x1, x2))


class SquareRoot(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return 1 / (2 * np.sqrt(x.value))  # TODO fix typing issues

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sqrt(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Exponent(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return np.sqrt(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.exp(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return tuple([1. / x.value])

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm2(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial / (x.value * np.log(2))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log2(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Logarithm10(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial / (x.value * np.log(10))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log10(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Sin(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial * np.cos(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sin(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Cos(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return - x.partial * np.sin(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cos(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Tan(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial / np.cos(x.value) ** 2

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tan(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arcsin(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arcsin(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arccos(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return - x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arccos(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Arctan(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial / (1 + x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arctan(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Sinh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial * np.cosh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sinh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Cosh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial * np.sinh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cosh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))


class Tanh(ReverseUniOperation):
    @staticmethod
    def get_inputs_partials(x: AbstractSpecialVariable) -> tuple[float]:
        return x.partial * (1 - np.tanh(x.value) ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tanh(x.value)
        return special_vars.Operator(value=value, inputs=tuple([x]))

