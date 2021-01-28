import numpy as np

from nnlibrary.nn.AbstractNeuralNetwork import AbstractNeuralNetwork
from nnlibrary.optimizers import Optimizer
from nnlibrary.structure import Structure
from nnlibrary.losses import Loss


class NeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, structure: Structure, loss: Loss, optimizer: Optimizer):
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer

    def feedforward(self, batch_data: np.ndarray):
        data = batch_data.copy()

        z_array = list()
        a_array = list()

        a_array.append(data.copy())

        for i in range(self.structure.layer_count - 1):
            prev_node, next_node = self.structure.node_counts[i], self.structure.node_counts[i + 1]
            s_pos, w_end, b_end = self.structure.variables_map[i]

            c_weight = self.structure.variables[s_pos:w_end].reshape((prev_node, next_node))
            c_bias = self.structure.variables[w_end:b_end]
            activation_func = self.structure.activations_function[i]

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

        pos, w_end, b_end = self.structure.variables_map[-1]
        gradient = np.zeros(b_end)

        cur_z = z_array[-1]
        prev_a, cur_a = a_array[-2], a_array[-1]

        delta = self.loss.derivative(cur_a, batch_target) * self.structure.activations_derivative[-1](cur_z)

        d_bias = np.mean(delta, axis=0)
        d_weight = np.dot(prev_a.T, delta).reshape((w_end - pos))

        gradient[pos:w_end] = d_weight
        gradient[w_end:b_end] = d_bias

        for i in range(2, self.structure.layer_count):
            next_node = self.structure.node_counts[-i]
            next_next_node = self.structure.node_counts[-(i - 1)]

            next_pos, next_w_end, _ = self.structure.variables_map[-(i - 1)]
            pos, w_end, b_end = self.structure.variables_map[-i]

            next_weight = self.structure.variables[next_pos:next_w_end].reshape((next_node, next_next_node))

            cur_z = z_array[-i]
            prev_cur_a = a_array[-(i + 1)]

            delta = np.dot(delta, next_weight.T) * self.structure.activations_derivative[-i](cur_z)

            d_bias = np.mean(delta, axis=0)
            d_weight = np.dot(prev_cur_a.T, delta).reshape((w_end - pos))

            gradient[pos:w_end] = d_weight
            gradient[w_end:b_end] = d_bias

        self.optimizer.optimize(training_variables=self.structure.variables, gradient_vector=gradient)
