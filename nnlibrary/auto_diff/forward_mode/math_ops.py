import numpy as np

from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff import UniOperation
from nnlibrary.auto_diff import BiOperation

from nnlibrary.auto_diff.forward_mode import special_vars


class Addition(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value + x2.value
        partial = x1.partial + x2.partial
        return special_vars.Variable(value=value, partial=partial)


class Subtraction(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value - x2.value
        partial = x1.partial - x2.partial
        return special_vars.Variable(value=value, partial=partial)


class Negative(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = -1 * x.value
        partial = -1 * x.partial
        return special_vars.Variable(value=value, partial=partial)


class Positive(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = +1 * x.value
        partial = +1 * x.partial
        return special_vars.Variable(value=value, partial=partial)


class Multiplication(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value * x2.value
        partial = x1.partial * x2.value + x1.value * x2.partial
        return special_vars.Variable(value=value, partial=partial)


class Division(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = x1.value / x2.value
        partial = (x1.partial * x2.value - x1.value * x2.partial) / x2.value ** 2
        return special_vars.Variable(value=value, partial=partial)


class Power(BiOperation):
    def __call__(self, x1: AbstractSpecialVariable, x2: AbstractSpecialVariable) -> AbstractSpecialVariable:
        log_x1 = np.log(self.epsilon) if abs(x1.value) < self.epsilon else np.log(x1.value)
        value = x1.value ** x2.value
        partial = x1.value ** (x2.value - 1) * (x1.partial * x2.value + x1.value * x2.partial * log_x1)
        return special_vars.Variable(value=value, partial=partial)


class SquareRoot(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sqrt(x.value)
        partial = x.partial / (2 * value)
        return special_vars.Variable(value=value, partial=partial)


class Exponent(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.exp(x.value)
        partial = x.partial * value
        return special_vars.Variable(value=value, partial=partial)


class Logarithm(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log(x.value)
        partial = x.partial / x.value
        return special_vars.Variable(value=value, partial=partial)


class Logarithm2(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log2(x.value)
        partial = x.partial / (x.value * np.log(2))
        return special_vars.Variable(value=value, partial=partial)


class Logarithm10(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.log10(x.value)
        partial = x.partial / (x.value * np.log(10))
        return special_vars.Variable(value=value, partial=partial)


class Sin(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sin(x.value)
        partial = x.partial * np.cos(x.value)
        return special_vars.Variable(value=value, partial=partial)


class Cos(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cos(x.value)
        partial = - x.partial * np.sin(x.value)
        return special_vars.Variable(value=value, partial=partial)


class Tan(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tan(x.value)
        partial = x.partial / np.cos(x.value) ** 2
        return special_vars.Variable(value=value, partial=partial)


class Arcsin(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arcsin(x.value)
        partial = x.partial / np.sqrt(1 - x.value ** 2)
        return special_vars.Variable(value=value, partial=partial)


class Arccos(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arccos(x.value)
        partial = - x.partial / np.sqrt(1 - x.value ** 2)
        return special_vars.Variable(value=value, partial=partial)


class Arctan(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.arctan(x.value)
        partial = x.partial / (1 + x.value ** 2)
        return special_vars.Variable(value=value, partial=partial)


class Sinh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.sinh(x.value)
        partial = x.partial * np.cosh(x.value)
        return special_vars.Variable(value=value, partial=partial)


class Cosh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.cosh(x.value)
        partial = x.partial * np.sinh(x.value)
        return special_vars.Variable(value=value, partial=partial)


class Tanh(UniOperation):
    def __call__(self, x: AbstractSpecialVariable) -> AbstractSpecialVariable:
        value = np.tanh(x.value)
        partial = x.partial * (1 - value ** 2)
        return special_vars.Variable(value=value, partial=partial)
