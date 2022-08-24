import numpy as np

from nnlibrary.auto_diff.variables import AbstractVariable
from nnlibrary.auto_diff.variables import Variable
from nnlibrary.auto_diff.math_ops import UniOperation
from nnlibrary.auto_diff.math_ops import BiOperation


class Addition(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value + x2.value
        gradient = x1.gradient + x2.gradient
        return Variable(value=value, gradient=gradient)


class Subtraction(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value - x2.value
        gradient = x1.gradient - x2.gradient
        return Variable(value=value, gradient=gradient)


class Negative(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = -1 * x.value
        gradient = -1 * x.gradient
        return Variable(value=value, gradient=gradient)


class Positive(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = +1 * x.value
        gradient = +1 * x.gradient
        return Variable(value=value, gradient=gradient)


class Multiplication(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value * x2.value
        gradient = x1.gradient * x2.value + x1.value * x2.gradient
        return Variable(value=value, gradient=gradient)


class Division(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value / x2.value
        gradient = (x1.gradient * x2.value - x1.value * x2.gradient) / x2.value ** 2
        return Variable(value=value, gradient=gradient)


class Power(BiOperation):
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        value = x1.value ** x2.value
        gradient = value * (x1.gradient * x2.value / x1.value + x2.gradient * np.log(x1.value))
        return Variable(value=value, gradient=gradient)


class Exponent(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.exp(x.value)
        gradient = value * x.gradient
        return Variable(value=value, gradient=gradient)


class Logarithm(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log(x.value)
        gradient = x.gradient / x.value
        return Variable(value=value, gradient=gradient)


class Logarithm2(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log2(x.value)
        gradient = x.gradient / (x.value * np.log(2))
        return Variable(value=value, gradient=gradient)


class Logarithm10(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.log10(x.value)
        gradient = x.gradient / (x.value * np.log(10))
        return Variable(value=value, gradient=gradient)


class Sin(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.sin(x.value)
        gradient = x.gradient * np.cos(x.value)
        return Variable(value=value, gradient=gradient)


class Cos(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.cos(x.value)
        gradient = - x.gradient * np.sin(x.value)
        return Variable(value=value, gradient=gradient)


class Tan(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.tan(x.value)
        gradient = x.gradient / np.cos(x.value) ** 2
        return Variable(value=value, gradient=gradient)


class Arcsin(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arcsin(x.value)
        gradient = x.gradient / np.sqrt(1 - x.value ** 2)
        return Variable(value=value, gradient=gradient)


class Arccos(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arccos(x.value)
        gradient = - x.gradient / np.sqrt(1 - x.value ** 2)
        return Variable(value=value, gradient=gradient)


class Arctan(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.arctan(x.value)
        gradient = x.gradient / (1 + x.value ** 2)
        return Variable(value=value, gradient=gradient)


class Sinh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.sinh(x.value)
        gradient = x.gradient * np.cosh(x.value)
        return Variable(value=value, gradient=gradient)


class Cosh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.cosh(x.value)
        gradient = x.gradient * np.sinh(x.value)
        return Variable(value=value, gradient=gradient)


class Tanh(UniOperation):
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        value = np.tanh(x.value)
        gradient = x.gradient * (1 - value ** 2)
        return Variable(value=value, gradient=gradient)
