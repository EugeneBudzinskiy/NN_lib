import numpy as np

from typing import Callable
from typing import Union

from nnlibrary.auto_diff_fast import AbstractNode
from nnlibrary.auto_diff_fast import AbstractMode
from nnlibrary.auto_diff_fast.forward_mode import special_vars


class ForwardMode(AbstractMode):
    def gradient(self,
                 func: Callable[[Union[np.ndarray, AbstractNode]], Union[np.ndarray, AbstractNode]],
                 x: np.ndarray) -> np.ndarray:

        var_x = special_vars.Node(values=x)
        result = np.empty_like(x)
        for i in range(result.shape[-1]):
            var_x.partials[:, i] = 1.
            result[:, i] = func(var_x).partials[:, i]
            var_x.partials[:, i] = 0.

        return result

