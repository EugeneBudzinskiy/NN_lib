import numpy as np

from typing import Callable
from typing import Union

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast import AbstractMode
from nnlibrary.auto_diff_fast.forward_mode import special_vars


class ForwardMode(AbstractMode):
    @staticmethod
    def jacobian(func: Callable[[Union[np.ndarray, AbstractNode]], Union[np.ndarray, AbstractNode]],
                 x: np.ndarray) -> np.ndarray:
        input_shape, output_shape = x.shape, func(x).shape
        var_x = special_vars.Node(values=x)

        result = np.zeros((np.prod(input_shape), np.prod(output_shape)))
        offset = output_shape[-1] - input_shape[-1]
        for i in range(x.shape[-1]):
            var_x.partials[:, i] = 1.

            p = func(var_x).partials
            for j in range(p.shape[-2]):
                result[j * (p.shape[-1] - offset) + i, j * p.shape[-1]:(j + 1) * p.shape[-1]] = p[j]

            var_x.partials[:, i] = 0.
        return result

