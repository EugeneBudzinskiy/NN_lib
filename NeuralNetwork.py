import numpy as np
from abc import ABC, abstractmethod


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def predict(self, x_data):
        pass

    @abstractmethod
    def learn(self, x_data, y_target):
        pass


class NeuralNetwork(AbstractNeuralNetwork):
    def __init__(self,
                 variables: np.ndarray,
                 var_map: tuple,
                 node_count: tuple,
                 activation_func: tuple,
                 activation_func_der: tuple,
                 loss_funcs: tuple,
                 optimizer,
                 layer_count):

        self.variables = variables
        self.var_map = var_map

        self.node_count = node_count
        self.act_function = activation_func
        self.act_function_der = activation_func_der

        self.loss, self.loss_der = loss_funcs
        self.optimizer = optimizer

        self.layer_count = layer_count

    @staticmethod
    def __check_input_data(x_data):
        if isinstance(x_data, np.ndarray):
            return x_data
        else:
            return np.array(x_data)

    def predict(self, x_data):
        current_data = self.__check_input_data(x_data)

        for i in range(self.layer_count):
            pass

    def learn(self, x_data, y_target):
        pass
