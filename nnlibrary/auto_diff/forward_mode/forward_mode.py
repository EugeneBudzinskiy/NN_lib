from typing import Callable

import numpy as np

from nnlibrary.auto_diff import AbstractMode
from . import special_vars


class ForwardMode(AbstractMode):
    @staticmethod
    def to_variable(x: np.ndarray) -> np.ndarray:
        vec_f = np.vectorize(lambda val: special_vars.Variable(value=val))
        return vec_f(val=x)

    @staticmethod
    def to_variable_direction(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        vec_f = np.vectorize(lambda val, grad: special_vars.Variable(value=val, partial=grad))
        return vec_f(val=x, grad=vector)

    def derivative(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        var_x = self.to_variable_direction(x=x, vector=np.ones_like(x))
        return self.partial_to_numpy(x=func(var_x))

    def gradient(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        var_x = self.to_variable(x=x)
        result = np.empty_like(x)
        for i in range(result.shape[-1]):
            self.set_partial(var_x=var_x[:, i], value=1.)
            result[:, i] = self.partial_to_numpy(x=func(var_x))
            self.set_partial(var_x=var_x[:, i], value=0.)

        return result

    def jacobian(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        var_x = self.to_variable(x=x)

        # First iteration
        self.set_partial(var_x=var_x[:, 0], value=1.)
        temp = self.partial_to_numpy(x=func(var_x))
        self.set_partial(var_x=var_x[:, 0], value=0.)

        # Creating `result` array only after knowing dimension of `func` output
        result = np.empty((temp.shape[-1], x.shape[-1]))
        result[:, 0] = temp

        for i in range(1, result.shape[-1]):
            self.set_partial(var_x=var_x[:, i], value=1.)
            result[:, i] = self.partial_to_numpy(x=func(var_x))
            self.set_partial(var_x=var_x[:, i], value=0.)

        return result

    def jvp(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        var_x = self.to_variable_direction(x=x, vector=vector)
        return self.partial_to_numpy(x=func(var_x))
