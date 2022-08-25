import numpy as np

from nnlibrary.auto_diff.variables import AbstractVariable
from nnlibrary.auto_diff.variables import Variable
from nnlibrary.auto_diff.math_ops import UniOperation
from nnlibrary.auto_diff.math_ops import BiOperation


class Addition(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value + x2.value
        gradient = x1.partial + x2.partial
        return Variable(value=value, partial=gradient)


class Subtraction(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value - x2.value
        gradient = x1.partial - x2.partial
        return Variable(value=value, partial=gradient)


class Negative(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = -1 * x.value
        gradient = -1 * x.partial
        return Variable(value=value, partial=gradient)


class Positive(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = +1 * x.value
        gradient = +1 * x.partial
        return Variable(value=value, partial=gradient)


class Multiplication(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value * x2.value
        gradient = x1.partial * x2.value + x1.value * x2.partial
        return Variable(value=value, partial=gradient)


class Division(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value / x2.value
        gradient = (x1.partial * x2.value - x1.value * x2.partial) / x2.value ** 2
        return Variable(value=value, partial=gradient)


class Power(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value ** x2.value
        gradient = value * (x1.partial * x2.value / x1.value + x2.partial * np.log(x1.value))
        return Variable(value=value, partial=gradient)


class SquareRoot(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.sqrt(x.value)
        gradient = x.partial / (2 * value)
        return Variable(value=value, partial=gradient)


class Exponent(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.exp(x.value)
        gradient = x.partial * value
        return Variable(value=value, partial=gradient)


class Logarithm(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log(x.value)
        gradient = x.partial / x.value
        return Variable(value=value, partial=gradient)


class Logarithm2(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log2(x.value)
        gradient = x.partial / (x.value * np.log(2))
        return Variable(value=value, partial=gradient)


class Logarithm10(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log10(x.value)
        gradient = x.partial / (x.value * np.log(10))
        return Variable(value=value, partial=gradient)


class Sin(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.sin(x.value)
        gradient = x.partial * np.cos(x.value)
        return Variable(value=value, partial=gradient)


class Cos(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.cos(x.value)
        gradient = - x.partial * np.sin(x.value)
        return Variable(value=value, partial=gradient)


class Tan(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.tan(x.value)
        gradient = x.partial / np.cos(x.value) ** 2
        return Variable(value=value, partial=gradient)


class Arcsin(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arcsin(x.value)
        gradient = x.partial / np.sqrt(1 - x.value ** 2)
        return Variable(value=value, partial=gradient)


class Arccos(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arccos(x.value)
        gradient = - x.partial / np.sqrt(1 - x.value ** 2)
        return Variable(value=value, partial=gradient)


class Arctan(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arctan(x.value)
        gradient = x.partial / (1 + x.value ** 2)
        return Variable(value=value, partial=gradient)


class Sinh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.sinh(x.value)
        gradient = x.partial * np.cosh(x.value)
        return Variable(value=value, partial=gradient)


class Cosh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.cosh(x.value)
        gradient = x.partial * np.sinh(x.value)
        return Variable(value=value, partial=gradient)


class Tanh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.tanh(x.value)
        gradient = x.partial * (1 - value ** 2)
        return Variable(value=value, partial=gradient)
