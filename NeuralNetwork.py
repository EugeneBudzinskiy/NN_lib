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
                 optimizer: id,
                 layer_count: int):

        self.variables = variables
        self.var_map = var_map

        self.node_count = node_count
        self.activation_func = activation_func
        self.activation_func_der = activation_func_der

        self.loss, self.loss_der = loss_funcs
        self.optimizer = optimizer

        self.layer_count = layer_count
        self.learning_rate = 0.05

    @staticmethod
    def __check_input_data(x_data):
        if isinstance(x_data, np.ndarray):
            return x_data.copy()
        else:
            return np.array([x_data])

    def predict(self, x_data):
        c_data = self.__check_input_data(x_data)

        for i in range(self.layer_count - 1):
            prev_nc = self.node_count[i]
            next_nc = self.node_count[i + 1]
            start, end_w, end_b = self.var_map[i]

            c_weights = self.variables[start:end_w].reshape(prev_nc, next_nc)
            c_biases = self.variables[end_w:end_b]
            c_activation_func = self.activation_func[i + 1]

            c_data = c_activation_func(np.dot(c_data, c_weights) + c_biases)

        return c_data

    def __feedforward_damp(self, x_data):
        a_buffer = list()
        z_buffer = list()

        c_data = self.__check_input_data(x_data)

        for i in range(self.layer_count - 1):
            prev_nc = self.node_count[i]
            next_nc = self.node_count[i + 1]
            start, end_w, end_b = self.var_map[i]

            c_weights = self.variables[start:end_w].reshape(prev_nc, next_nc)
            c_biases = self.variables[end_w:end_b]
            c_activation_func = self.activation_func[i + 1]

            non_act = np.dot(c_data, c_weights) + c_biases
            z_buffer.append(non_act.copy())

            c_data = c_activation_func(non_act)
            a_buffer.append(c_data)

        return tuple(a_buffer), tuple(z_buffer)

    def learn(self, x_data, y_target):
        a_buffer, z_buffer = self.__feedforward_damp(x_data)
        start, w_end, b_end = self.var_map[-1]

        gradient = np.zeros(b_end)

        biases_delta = self.loss_der(a_buffer[-1], y_target) * self.activation_func_der[-1](z_buffer[-1])
        weights_delta = np.dot(a_buffer[-2].T, biases_delta)

        gradient[start:w_end] = weights_delta.reshape(w_end - start)
        gradient[w_end:b_end] = biases_delta.reshape(b_end - w_end)

        for i in range(2, self.layer_count - 1):
            prev_nc = self.node_count[-(i - 1)]
            next_nc = self.node_count[-i]

            start, w_end, b_end = self.var_map[-i]
            next_start, next_w_end, next_b_end = self.var_map[-(i - 1)]

            next_weights = self.variables[next_start:next_w_end].reshape(prev_nc, next_nc)
            biases_delta = np.dot(biases_delta, next_weights) * self.activation_func_der[-i](z_buffer[-i])
            weights_delta = np.dot(a_buffer[-(i + 1)].T, biases_delta)

            gradient[start:w_end] = weights_delta.reshape(w_end - start)
            gradient[w_end:b_end] = biases_delta.reshape(b_end - w_end)

        # TODO make optimizer for gradient

        return gradient

    def batch_learning(self, x_data_batch, y_target_batch):
        _, _, size = self.var_map[-1]
        batch_gradient = np.zeros(size)
        ln = len(x_data_batch)

        for i in range(ln):
            batch_gradient += self.learn(x_data_batch[i], y_target_batch[i])

        batch_gradient /= ln
        batch_gradient *= self.learning_rate

        self.variables -= batch_gradient
