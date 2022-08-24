from nnlibrary.auto_diff.variables import AbstractVariable


class Variable(AbstractVariable):
    def __init__(self, value: float, gradient: float = 0.):
        super(Variable, self).__init__(value=value, gradient=gradient)

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __neg__(self):
        pass

    def __pos__(self):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __pow__(self, power, modulo=None):
        pass
