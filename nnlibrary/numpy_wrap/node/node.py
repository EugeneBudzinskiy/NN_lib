import logging
from typing import Union

from nnlibrary import numpy_wrap as npw
from nnlibrary.numpy_wrap import node_ops
from .abstractions import AbstractNode


class Node(AbstractNode):
    @staticmethod
    def _wrapper(other):
        return other if isinstance(other, AbstractNode) else Node(other)

    @staticmethod
    def unwrap_if_needed(array: Union[npw.ndarray, AbstractNode], verbose: bool = True) -> AbstractNode:
        if isinstance(array, npw.ndarray):
            if verbose:
                logging.warning(msg='Inefficient operation was used!')

            flat, ln = array.flatten(), array.size
            values, partials = npw.numpy.zeros(ln), npw.numpy.zeros(ln)
            for i in range(ln):
                values[i] = flat[i].values
                partials[i] = flat[i].partials
            return Node(values=values.reshape(array.shape), partials=partials.reshape(array.shape))

        return array

    def __repr__(self):
        return f'{self.values}'

    def __getitem__(self, item):
        return Node(values=self.values[item], partials=self.partials[item])

    def __len__(self):
        return len(self.values)

    def __add__(self, other):
        return node_ops.Addition.call(x1=self, x2=self._wrapper(other=other))

    def __radd__(self, other):
        return node_ops.Addition.call(x1=self._wrapper(other=other), x2=self)

    def __sub__(self, other):
        return node_ops.Subtraction.call(x1=self, x2=self._wrapper(other=other))

    def __rsub__(self, other):
        return node_ops.Subtraction.call(x1=self._wrapper(other=other), x2=self)

    def __mul__(self, other):
        return node_ops.Multiplication.call(x1=self, x2=self._wrapper(other=other))

    def __rmul__(self, other):
        return node_ops.Multiplication.call(x1=self._wrapper(other=other), x2=self)

    def __truediv__(self, other):
        return node_ops.Division.call(x1=self, x2=self._wrapper(other=other))

    def __rtruediv__(self, other):
        return node_ops.Division.call(x1=self._wrapper(other=other), x2=self)

    def __pow__(self, power, modulo=None):
        return node_ops.Power.call(x1=self, x2=self._wrapper(other=power))

    def __rpow__(self, power, modulo=None):
        return node_ops.Power.call(x1=self._wrapper(other=power), x2=self)

    def sqrt(self):
        return node_ops.SquareRoot.call(x=self)

    def exp(self):
        return node_ops.Exponent.call(x=self)

    def log(self):
        return node_ops.Logarithm.call(x=self)

    def log2(self):
        return node_ops.Logarithm2.call(x=self)

    def log10(self):
        return node_ops.Logarithm10.call(x=self)

    def sin(self):
        return node_ops.Sin.call(x=self)

    def cos(self):
        return node_ops.Cos.call(x=self)

    def tan(self):
        return node_ops.Tan.call(x=self)

    def arcsin(self):
        return node_ops.Arcsin.call(x=self)

    def arccos(self):
        return node_ops.Arccos.call(x=self)

    def arctan(self):
        return node_ops.Arctan.call(x=self)

    def sinh(self):
        return node_ops.Sinh.call(x=self)

    def cosh(self):
        return node_ops.Cosh.call(x=self)

    def tanh(self):
        return node_ops.Tanh.call(x=self)

    def __matmul__(self, other):
        return node_ops.MatrixMultiplication.call(x1=self, x2=self._wrapper(other=other))

    def __rmatmul__(self, other):
        return node_ops.MatrixMultiplication.call(x1=self._wrapper(other=other), x2=self)

    def sum(self, *args, **kwargs):
        return node_ops.Summation.call(x=self, *args, **kwargs)

    def __abs__(self):
        return node_ops.Absolute.call(x=self)

    def __neg__(self):
        return node_ops.Negative.call(x=self)

    def __pos__(self):
        return node_ops.Positive.call(x=self)

    def __eq__(self, other):
        return self.values == self._wrapper(other=other).values

    def __ne__(self, other):
        return self.values != self._wrapper(other=other).values

    def __le__(self, other):
        return self.values <= self._wrapper(other=other).values

    def __ge__(self, other):
        return self.values >= self._wrapper(other=other).values

    def __lt__(self, other):
        return self.values < self._wrapper(other=other).values

    def __gt__(self, other):
        return self.values > self._wrapper(other=other).values

