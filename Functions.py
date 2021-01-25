import numpy as np


class Func:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.__act_func_der_dict = {
            self.sigmoid: self.sigmoid_der,
            self.relu: self.relu_der,
            self.linear: self.linear_der
        }
        self.__loss_func_der_dict = {
            self.mse: self.mse_der
        }

    def get_act_func_der(self, act_function):
        return self.__act_func_der_dict[act_function]

    def get_loss_func_der(self, loss_function):
        return self.__loss_func_der_dict[loss_function]

    """ ACTIVATION FUNCTIONS """
    @staticmethod
    def linear(x):
        return x.copy()

    @staticmethod
    def linear_der(x):
        return 1

    @staticmethod
    def relu(x):
        z = x.copy()
        z[z < 0] = 0
        return z

    @staticmethod
    def relu_der(x):
        z = x.copy()
        z[z < 0] = 0
        z[z > 0] = 1
        return z

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        z = self.sigmoid(x)
        return z * (1 - z)

    @staticmethod
    def softmax(x):
        z = np.exp(x)
        return z / np.sum(z)

    @staticmethod
    def softmax_der(x):
        shape = x.shape
        s = x.reshape(-1, 1)
        d = np.diagflat(s) - np.dot(s, s.T)
        return d.diagonal().reshape(shape)

    """ LOSS FUNCTIONS """
    @staticmethod
    def mse(y, y_pred):
        return np.mean(np.square(y - y_pred))

    @staticmethod
    def mse_der(y, y_pred):
        return 2 * (y - y_pred)

    """ OPTIMIZER FUNCTIONS """
    @staticmethod
    def adam(x):
        return x
