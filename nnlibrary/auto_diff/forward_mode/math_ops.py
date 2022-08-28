import numpy as np

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import UniOperation
from nnlibrary.auto_diff import BiOperation

from nnlibrary.auto_diff.forward_mode import special_vars


class Addition(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial + x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value + x2.value
        partial = Addition.partial(x1=x1, x2=x2)
        return special_vars.Variable(value=value, partial=partial)


class Subtraction(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial - x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value - x2.value
        partial = Subtraction.partial(x1=x1, x2=x2)
        return special_vars.Variable(value=value, partial=partial)


class Negative(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return -1 * x.partial

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = -1 * x.value
        partial = Negative.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Positive(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return +1 * x.partial

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = +1 * x.value
        partial = Positive.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Multiplication(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return x1.partial * x2.value + x1.value * x2.partial

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value * x2.value
        partial = Multiplication.partial(x1=x1, x2=x2)
        return special_vars.Variable(value=value, partial=partial)


class Division(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        return (x1.partial * x2.value - x1.value * x2.partial) / x2.value ** 2

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value / x2.value
        partial = Division.partial(x1=x1, x2=x2)
        return special_vars.Variable(value=value, partial=partial)


class Power(BiOperation):
    @staticmethod
    def partial(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> float:
        log_x1 = np.log(Power.epsilon) if abs(x1.value) < Power.epsilon else np.log(x1.value)
        return x1.value ** (x2.value - 1) * (x1.partial * x2.value + x1.value * x2.partial * log_x1)

    @staticmethod
    def call(x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value ** x2.value
        partial = Power.partial(x1=x1, x2=x2)
        return special_vars.Variable(value=value, partial=partial)


class SquareRoot(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (2 * np.sqrt(x.value))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sqrt(x.value)
        partial = SquareRoot.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Exponent(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.exp(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.exp(x.value)
        partial = Exponent.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Logarithm(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / x.value

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log(x.value)
        partial = Logarithm.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Logarithm2(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (x.value * np.log(2))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log2(x.value)
        partial = Logarithm2.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Logarithm10(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (x.value * np.log(10))

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log10(x.value)
        partial = Logarithm10.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Sin(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.cos(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sin(x.value)
        partial = Sin.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Cos(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return - x.partial * np.sin(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cos(x.value)
        partial = Cos.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Tan(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / np.cos(x.value) ** 2

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tan(x.value)
        partial = Tan.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Arcsin(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arcsin(x.value)
        partial = Arcsin.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Arccos(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return - x.partial / np.sqrt(1 - x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arccos(x.value)
        partial = Arccos.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Arctan(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial / (1 + x.value ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arctan(x.value)
        partial = Arctan.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Sinh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.cosh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sinh(x.value)
        partial = Sinh.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Cosh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * np.sinh(x.value)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cosh(x.value)
        partial = Cosh.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)


class Tanh(UniOperation):
    @staticmethod
    def partial(x: AbstractSpecialVariable) -> float:
        return x.partial * (1 - np.tanh(x.value) ** 2)

    @staticmethod
    def call(x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tanh(x.value)
        partial = Tanh.partial(x=x)
        return special_vars.Variable(value=value, partial=partial)
