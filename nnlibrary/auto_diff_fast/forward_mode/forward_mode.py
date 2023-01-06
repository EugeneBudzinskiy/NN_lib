from typing import Any
from typing import Callable
from typing import Union

import numpy as np

from nnlibrary.auto_diff_fast import AbstractMode
from .abstractions import AbstractNode
from .special_vars import Node


class ForwardMode(AbstractMode):
    @staticmethod
    def wrapper(func_output: Union[np.ndarray, AbstractNode]) -> Callable[[Any], AbstractNode]:
        def helper(f_out: np.ndarray, t: np.ndarray):
            f_out = f_out.reshape((1, -1)) if f_out.ndim == 1 else f_out
            values, partials = np.zeros_like(f_out), np.zeros_like(f_out)
            for i in range(f_out.shape[-2]):
                for j in range(f_out.shape[-1]):
                    values[i, j] = t[i, j].values
                    partials[i, j] = t[i, j].partials

            return Node(values=values, partials=partials)

        if isinstance(func_output, AbstractNode):
            return lambda t: t

        return lambda t: helper(f_out=func_output, t=t)

    @staticmethod
    def jacobian(func: Callable[[Union[np.ndarray, AbstractNode]], Union[np.ndarray, AbstractNode]],
                 x: np.ndarray) -> np.ndarray:
        input_shape, output_shape = x.shape, func(x).shape
        var_x = Node(values=x)

        result = np.zeros((np.prod(input_shape), np.prod(output_shape)))
        offset = output_shape[-1] - input_shape[-1]
        for i in range(x.shape[-1]):
            var_x.partials[:, i] = 1.

            p = Node.unwrap_if_needed(func(var_x)).partials
            for j in range(p.shape[-2]):
                result[j * (p.shape[-1] - offset) + i, j * p.shape[-1]:(j + 1) * p.shape[-1]] = p[j]

            var_x.partials[:, i] = 0.
        return result.T

