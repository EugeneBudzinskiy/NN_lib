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
        gradient = value * (x1.gradient * x2.value / x1.value + x2.gradient * np.log(x1))
        return Variable(value=value, gradient=gradient)
