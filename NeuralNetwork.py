import numpy as np
from abc import ABC, abstractmethod


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def predict(self, batch_data):
        pass

    @abstractmethod
    def learn(self, batch_data, batch_target):
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

        self.learning_rate = 1

    def feedforward(self, batch_data: np.ndarray):
        data = batch_data.copy()

        z_array = list()
        a_array = list()

        a_array.append(data.copy())

        for i in range(self.layer_count - 1):
            prev_node, next_node = self.node_count[i], self.node_count[i + 1]
            s_pos, w_end, b_end = self.var_map[i]

            c_weight = self.variables[s_pos:w_end].reshape((prev_node, next_node))
            c_bias = self.variables[w_end:b_end]
            activation_func = self.activation_func[i]

            z = np.dot(data, c_weight) + c_bias
            z_array.append(z.copy())

            data = activation_func(z)
            a_array.append(data.copy())

        return tuple(z_array), tuple(a_array)

    def predict(self, batch_data: np.ndarray):
        _, result = self.feedforward(batch_data)
        return result[-1]

    def learn(self, batch_data: np.ndarray, batch_target: np.ndarray):
        z_array, a_array = self.feedforward(batch_data)

        pos, w_end, b_end = self.var_map[-1]
        gradient = np.zeros(b_end)

        cur_z = z_array[-1]
        prev_a, cur_a = a_array[-2], a_array[-1]

        print(self.loss(cur_a, batch_target))

        delta = self.loss_der(cur_a, batch_target) * self.activation_func_der[-1](cur_z)

        d_bias = np.mean(delta, axis=0)
        d_weight = np.dot(prev_a.T, delta).reshape((w_end - pos))

        gradient[pos:w_end] = d_weight
        gradient[w_end:b_end] = d_bias

        for i in range(2, self.layer_count):
            next_node = self.node_count[-i]
            next_next_node = self.node_count[-(i - 1)]

            next_pos, next_w_end, _ = self.var_map[-(i - 1)]
            pos, w_end, b_end = self.var_map[-i]

            next_weight = self.variables[next_pos:next_w_end].reshape((next_node, next_next_node))

            cur_z = z_array[-i]
            prev_cur_a = a_array[-(i + 1)]

            delta = np.dot(delta, next_weight.T) * self.activation_func_der[-i](cur_z)

            d_bias = np.mean(delta, axis=0)
            d_weight = np.dot(prev_cur_a.T, delta).reshape((w_end - pos))

            gradient[pos:w_end] = d_weight
            gradient[w_end:b_end] = d_bias

        # TODO Realization of gradient optimizer

        self.variables += self.learning_rate * gradient
