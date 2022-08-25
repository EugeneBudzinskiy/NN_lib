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

    def exp(self):
        return math_ops.Exponent().__call__(x=self)

    def log(self):
        return math_ops.Logarithm().__call__(x=self)

    def log2(self):
        return math_ops.Logarithm2().__call__(x=self)

    def log10(self):
        return math_ops.Logarithm10().__call__(x=self)

    def sin(self):
        return math_ops.Sin().__call__(x=self)

    def cos(self):
        return math_ops.Cos().__call__(x=self)

    def tan(self):
        return math_ops.Tan().__call__(x=self)

    def arcsin(self):
        return math_ops.Arcsin().__call__(x=self)

    def arccos(self):
        return math_ops.Arccos().__call__(x=self)

    def arctan(self):
        return math_ops.Arctan().__call__(x=self)

    def sinh(self):
        return math_ops.Sinh().__call__(x=self)

    def cosh(self):
        return math_ops.Cosh().__call__(x=self)

    def tanh(self):
        return math_ops.Tanh().__call__(x=self)
