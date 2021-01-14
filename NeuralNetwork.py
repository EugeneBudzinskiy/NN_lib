import numpy as np
from abc import ABC, abstractmethod


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def predict(self, x_data):
        pass

    @abstractmethod
    def learn(self, batch_data):
        pass


class NeuralNetwork(AbstractNeuralNetwork):
    def __init__(self,
                 variables: np.ndarray,
                 var_map: tuple,
                 node_count: tuple,
                 activation_func: tuple,
                 activation_func_der: tuple,
                 loss_func: id,
                 loss_func_der: id,
                 optimizer: id,
                 layer_count: int):

        self.variables = variables
        self.var_map = var_map
        self.node_count = node_count

        self.activation_func = activation_func
        self.activation_func_der = activation_func_der

        self.loss = loss_func
        self.loss_der = loss_func_der

        self.optimizer = optimizer
        self.layer_count = layer_count

        self.learning_rate = 0.05

    def feedforward(self, batch_data: np.ndarray):
        data = batch_data.copy()

        z_array = list()
        a_array = list()

        for i in range(self.layer_count - 1):
            node_prev = self.node_count[i]
            node_next = self.node_count[i + 1]

            s_pos, w_end, b_end = self.var_map[i]
            c_weight = self.variables[s_pos:w_end].reshape(node_prev, node_next)
            c_bias = self.variables[w_end:b_end]

            act_func = self.activation_func[i]

            z = np.dot(data, c_weight) + c_bias
            z_array.append(z.copy())

            data = act_func(z)
            a_array.append(data.copy())

        return tuple(z_array), tuple(a_array)

    def predict(self, batch_data: np.ndarray):
        _, result = self.feedforward(batch_data)
        return result[-1]

    def learn(self, batch_data):
        pass
