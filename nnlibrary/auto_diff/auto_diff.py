import numpy as np

from typing import Callable

from nnlibrary.auto_diff.variables import Variable


class AutoDiff:
    @staticmethod
    def to_variable(x: np.ndarray, vector: np.ndarray = None) -> np.ndarray:
        if vector is None:
            return np.vectorize(lambda val: Variable(value=val))(v=x)

        return np.vectorize(lambda val, grad: Variable(value=val, gradient=grad))(val=x, grad=vector)

    @staticmethod
    def gradient_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.gradient)(x)

    @staticmethod
    def value_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.value)(x)

    @staticmethod
    def jvp(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, vector: np.ndarray):
        var_x = AutoDiff.to_variable(x=x, vector=vector)
        return AutoDiff.gradient_to_numpy(x=func(var_x))
