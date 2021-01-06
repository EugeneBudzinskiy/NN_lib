import numpy as np


class Func:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    """ ACTIVATION FUNCTIONS """
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

    """ LOSS FUNCTIONS """
    @staticmethod
    def mse(y, y_pred):
        return np.mean((y - y_pred) ** 2) / 2

    @staticmethod
    def mse_der(y, y_pred):
        return np.mean(y - y_pred)

    """ OPTIMIZER FUNCTIONS """
    @staticmethod
    def adam(x):
        return x
