from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff.backward_mode import math_ops


class Variable(AbstractSpecialVariable):
    def __init__(self, value: float, partial: float = 0.):
        super(Variable, self).__init__(value=value, partial=partial)

    def _wrapper(self, other):
        return other if isinstance(other, AbstractSpecialVariable) else Operator(other)

    def __repr__(self):
        return self.value, self.partial

    def __add__(self, other):
        return math_ops.Addition().__call__(x1=self, x2=self._wrapper(other=other))

    def __sub__(self, other):
        return math_ops.Subtraction().__call__(x1=self, x2=self._wrapper(other=other))

    def __mul__(self, other):
        return math_ops.Multiplication().__call__(x1=self, x2=self._wrapper(other=other))

    def __truediv__(self, other):
        return math_ops.Division().__call__(x1=self, x2=self._wrapper(other=other))

    def __pow__(self, power, modulo=None):
        return math_ops.Power().__call__(x1=self, x2=self._wrapper(other=power))

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
        return math_ops.Negative().__call__(x=self)

    def __pos__(self):
        return math_ops.Positive().__call__(x=self)

    def sqrt(self):
        return math_ops.SquareRoot().__call__(x=self)

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


class Operator(Variable):
    def __init__(self, value: float, partial: float = 0.):
        super(Operator, self).__init__(value=value, partial=partial)
        self.inputs = tuple()
