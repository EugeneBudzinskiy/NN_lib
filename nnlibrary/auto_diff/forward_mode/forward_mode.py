import numpy as np

from typing import Callable

from nnlibrary.auto_diff.forward_mode import special_vars


class ForwardMode:
    @staticmethod
    def to_variable(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda val: special_vars.Variable(value=val))(val=x)

    @staticmethod
    def to_variable_direction(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda val, grad: special_vars.Variable(value=val, partial=grad))(val=x, grad=vector)

    @staticmethod
    def partial_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.get_inputs_partials)(x)

    @staticmethod
    def value_to_numpy(x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda v: v.value)(x)

    @staticmethod
    def set_partial(var_x: np.ndarray, value: float):
        for i in range(var_x.shape[-1]):
            var_x[i].get_inputs_partials = value

    def derivative(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
        var_x = self.to_variable_direction(x=x, vector=np.ones_like(x))
        return self.partial_to_numpy(x=func(var_x))

    def gradient(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
        var_x = self.to_variable(x=x)
        result = np.empty_like(x)
        for i in range(result.shape[-1]):
            self.set_partial(var_x=var_x[:, i], value=1.)
            result[:, i] = self.partial_to_numpy(x=func(var_x))
            self.set_partial(var_x=var_x[:, i], value=0.)

        return result

    def jacobian(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
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

    def jvp(self, func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, vector: np.ndarray):
        var_x = self.to_variable_direction(x=x, vector=vector)
        return self.partial_to_numpy(x=func(var_x))
