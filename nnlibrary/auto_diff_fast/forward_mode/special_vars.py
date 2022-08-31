import numpy as np

from nnlibrary.auto_diff_fast.forward_mode import math_ops

from nnlibrary.auto_diff_fast import AbstractNode


class Node(AbstractNode):
    def __repr__(self):
        return f'{self.values, self.partials}'

    @staticmethod
    def _wrapper(other):
        return other if isinstance(other, AbstractNode) else Node(other)

    def __add__(self, other):
        return math_ops.Addition.call(x1=self, x2=self._wrapper(other=other))

    def __radd__(self, other):
        return math_ops.Addition.call(x1=self._wrapper(other=other), x2=self)

    def __mul__(self, other):
        return math_ops.Multiplication.call(x1=self, x2=self._wrapper(other=other))

    def __rmul__(self, other):
        return math_ops.Multiplication.call(x1=self._wrapper(other=other), x2=self)

    def __matmul__(self, other):
        return math_ops.MatrixMultiplication.call(x1=self, x2=self._wrapper(other=other))

    def __rmatmul__(self, other):
        return math_ops.MatrixMultiplication.call(x1=self._wrapper(other=other), x2=self)

    def sum(self, *args, **kwargs):
        return math_ops.Summation.call(x=self, *args, **kwargs)


