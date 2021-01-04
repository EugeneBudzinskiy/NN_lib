import numpy as np


class Funcs:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.__activation_functions = {
            'linear': (
                self.linear,
                self.linear_der
            ),
            'relu': (
                self.relu,
                self.relu_der
            ),
            'sigmoid': (
                self.sigmoid,
                self.sigmoid_der
            )
        }

        self.__loss_functions = {
            'mse': (
                self.mse,
                self.mse_der
            )
        }

    """ FUNCTION GETTERS """
    def get_activation_function(self, act_func_name: str):
        if act_func_name in self.__activation_functions:
            return self.__activation_functions[act_func_name]
        raise NotImplementedError(
            f"The activation function '{act_func_name}' doesn't exist"  # TODO MB Rewrite Error
        )

    def get_loss_function(self, loss_func_name: str):
        if loss_func_name in self.__loss_functions:
            return self.__loss_functions[loss_func_name]
        raise NotImplementedError(
            f"The activation function '{loss_func_name}' doesn't exist"  # TODO MB Rewrite Error
        )

    """ ACTIVATION FUNCTION """
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_der(x):
        return x

    @staticmethod
    def relu(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def relu_der(x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        z = self.sigmoid(x)
        return z * (1 - z)

    """ LOSS FUNCTION """
    @staticmethod
    def mse(y, y_pred):
        return np.mean((y - y_pred) ** 2) / 2

    @staticmethod
    def mse_der(y, y_pred):
        return np.mean(y - y_pred)
