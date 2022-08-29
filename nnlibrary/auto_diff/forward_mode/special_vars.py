from nnlibrary.auto_diff.forward_mode import math_ops

from nnlibrary.auto_diff import AbstractSpecialVariable


class Variable(AbstractSpecialVariable):
    def __init__(self, value: float, partial: float = 0.):
        super(Variable, self).__init__(value=value, partial=partial)

    @staticmethod
    def _wrapper(other):
        return other if isinstance(other, AbstractSpecialVariable) else Variable(other)

    def __add__(self, other):
        return math_ops.Addition.call(x1=self, x2=self._wrapper(other=other))

    def __sub__(self, other):
        return math_ops.Subtraction.call(x1=self, x2=self._wrapper(other=other))

    def __mul__(self, other):
        return math_ops.Multiplication.call(x1=self, x2=self._wrapper(other=other))

    def __truediv__(self, other):
        return math_ops.Division.call(x1=self, x2=self._wrapper(other=other))

    def __pow__(self, power, modulo=None):
        return math_ops.Power.call(x1=self, x2=self._wrapper(other=power))

    def __eq__(self, other):
        return self.value == self._wrapper(other=other).value

    def __ne__(self, other):
        return self.value != self._wrapper(other=other).value

    def __le__(self, other):
        return self.value <= self._wrapper(other=other).value

    def __ge__(self, other):
        return self.value >= self._wrapper(other=other).value

    def __lt__(self, other):
        return self.value < self._wrapper(other=other).value

    def __gt__(self, other):
        return self.value > self._wrapper(other=other).value

    def __neg__(self):
        return math_ops.Negative.call(x=self)

    def __pos__(self):
        return math_ops.Positive.call(x=self)

    def sqrt(self):
        return math_ops.SquareRoot.call(x=self)

    def exp(self):
        return math_ops.Exponent.call(x=self)

    def log(self):
        return math_ops.Logarithm.call(x=self)

    def log2(self):
        return math_ops.Logarithm2.call(x=self)

    def log10(self):
        return math_ops.Logarithm10.call(x=self)

    def sin(self):
        return math_ops.Sin.call(x=self)

    def cos(self):
        return math_ops.Cos.call(x=self)

    def tan(self):
        return math_ops.Tan.call(x=self)

    def arcsin(self):
        return math_ops.Arcsin.call(x=self)

    def arccos(self):
        return math_ops.Arccos.call(x=self)

    def arctan(self):
        return math_ops.Arctan.call(x=self)

    def sinh(self):
        return math_ops.Sinh.call(x=self)

    def cosh(self):
        return math_ops.Cosh.call(x=self)

    def tanh(self):
        return math_ops.Tanh.call(x=self)

