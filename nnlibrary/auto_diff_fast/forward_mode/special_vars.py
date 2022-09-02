import logging
from typing import Union

import numpy as np

from . import math_ops

from .abstractions import AbstractNode


class Node(AbstractNode):
    @staticmethod
    def _wrapper(other):
        return other if isinstance(other, AbstractNode) else Node(other)

    @staticmethod
    def unwrap_if_needed(array: Union[np.ndarray, AbstractNode], verbose: bool = True) -> AbstractNode:

        if isinstance(array, np.ndarray):
            if verbose:
                logging.warning(msg='Inefficient operation was used!')

            flat, ln = array.flatten(), array.size
            values, partials = np.zeros(ln), np.zeros(ln)
            for i in range(ln):
                values[i] = flat[i].values
                partials[i] = flat[i].partials
            return Node(values=values.reshape(array.shape), partials=partials.reshape(array.shape))

        return array

    def __repr__(self):
        return f'{self.values, self.partials}'

    def __getitem__(self, item):
        return Node(values=self.values[item], partials=self.partials[item])

    def __len__(self):
        return len(self.values)

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

