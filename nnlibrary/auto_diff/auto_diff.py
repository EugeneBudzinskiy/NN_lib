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

    @staticmethod
    def gradient(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
        var_x = AutoDiff.to_variable(x=x)
        result = np.empty_like(x)
        for i in range(result.shape[-1]):
            AutoDiff.set_partial(var_x=var_x[:, i], value=1.)
            result[:, i] = AutoDiff.partial_to_numpy(x=func(var_x))
            AutoDiff.set_partial(var_x=var_x[:, i], value=0.)

        return result

    @staticmethod
    def jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
        var_x = AutoDiff.to_variable(x=x)

        # First iteration
        AutoDiff.set_partial(var_x=var_x[:, 0], value=1.)
        temp = AutoDiff.partial_to_numpy(x=func(var_x))
        AutoDiff.set_partial(var_x=var_x[:, 0], value=0.)

        # Creating `result` array only after knowing dimension of `func` output
        result = np.empty((temp.shape[-1], x.shape[-1]))
        result[:, 0] = temp

        for i in range(1, result.shape[-1]):
            AutoDiff.set_partial(var_x=var_x[:, i], value=1.)
            result[:, i] = AutoDiff.partial_to_numpy(x=func(var_x))
            AutoDiff.set_partial(var_x=var_x[:, i], value=0.)

        return result

    @staticmethod
    def jvp(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, vector: np.ndarray):
        var_x = AutoDiff.to_variable_direction(x=x, vector=vector)
        return AutoDiff.partial_to_numpy(x=func(var_x))
