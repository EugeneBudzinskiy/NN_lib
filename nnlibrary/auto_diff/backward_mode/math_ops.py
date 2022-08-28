import numpy as np

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import UniOperation
from nnlibrary.auto_diff import BiOperation

from nnlibrary.auto_diff.backward_mode import special_vars


class Addition(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value + x2.value
        return special_vars.Operator(value=value)


class Subtraction(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value - x2.value
        return special_vars.Operator(value=value)


class Negative(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = -1 * x.value
        return special_vars.Operator(value=value)


class Positive(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = +1 * x.value
        return special_vars.Operator(value=value)


class Multiplication(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value * x2.value
        return special_vars.Operator(value=value)


class Division(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value / x2.value
        return special_vars.Operator(value=value)


class Power(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value ** x2.value
        return special_vars.Operator(value=value)


class SquareRoot(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sqrt(x.value)
        return special_vars.Operator(value=value)


class Exponent(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.exp(x.value)
        return special_vars.Operator(value=value)


class Logarithm(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log(x.value)
        return special_vars.Operator(value=value)


class Logarithm2(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log2(x.value)
        return special_vars.Operator(value=value)


class Logarithm10(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log10(x.value)
        return special_vars.Operator(value=value)


class Sin(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sin(x.value)
        return special_vars.Operator(value=value)


class Cos(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cos(x.value)
        return special_vars.Operator(value=value)


class Tan(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tan(x.value)
        return special_vars.Operator(value=value)


class Arcsin(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arcsin(x.value)
        return special_vars.Operator(value=value)


class Arccos(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arccos(x.value)
        return special_vars.Operator(value=value)


class Arctan(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arctan(x.value)
        return special_vars.Operator(value=value)


class Sinh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sinh(x.value)
        return special_vars.Operator(value=value)


class Cosh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cosh(x.value)
        return special_vars.Operator(value=value)


class Tanh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tanh(x.value)
        return special_vars.Operator(value=value)
