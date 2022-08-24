from nnlibrary.auto_diff import math_ops

from nnlibrary.auto_diff.variables import AbstractVariable


class Variable(AbstractVariable):
    def __init__(self, value: float, gradient: float = 0.):
        super(Variable, self).__init__(value=value, gradient=gradient)

    def __add__(self, other):
        return math_ops.Addition().__call__(x1=self, x2=other)

    def __sub__(self, other):
        return math_ops.Subtraction().__call__(x1=self, x2=other)

    def __neg__(self):
        return math_ops.Negative().__call__(x=self)

    def __pos__(self):
        return math_ops.Positive().__call__(x=self)

    def __mul__(self, other):
        return math_ops.Multiplication().__call__(x1=self, x2=other)

    def __truediv__(self, other):
        return math_ops.Division().__call__(x1=self, x2=other)

    def __pow__(self, power, modulo=None):
        return math_ops.Power().__call__(x1=self, x2=power)
