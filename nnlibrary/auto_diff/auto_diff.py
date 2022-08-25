import numpy as np

from typing import Callable

from nnlibrary.auto_diff.variables import Variable


class AutoDiff:
    @staticmethod
    def to_variable(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda val: Variable(value=val))(val=x)

    @staticmethod
    def to_variable_direction(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda val, grad: Variable(value=val, partial=grad))(val=x, grad=vector)

    @staticmethod
    def partial_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.partial)(x)

    @staticmethod
    def value_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.value)(x)

    @staticmethod
    def set_partial(var_x: np.ndarray, value: float):
        for i in range(var_x.shape[-1]):
            var_x[i].partial = value

    @staticmethod
    def derivative(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
        var_x = AutoDiff.to_variable_direction(x=x, vector=np.ones_like(x))
        return AutoDiff.partial_to_numpy(x=func(var_x))

    # @staticmethod
    # def gradient(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    #     var_x = AutoDiff.to_variable(x=x)
    #     return AutoDiff.partial_to_numpy(x=func(var_x))

    @staticmethod
    def jvp(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, vector: np.ndarray):
        var_x = AutoDiff.to_variable_direction(x=x, vector=vector)
        return AutoDiff.partial_to_numpy(x=func(var_x))
