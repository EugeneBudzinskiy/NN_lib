import numpy as np

from nnlibrary.nn.AbstractNeuralNetwork import AbstractNeuralNetwork
from nnlibrary.optimizers import Optimizer
from nnlibrary.losses import Loss


class NeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, structure: tuple, loss: Loss, optimizer: Optimizer):
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer

        self.variable_map = self._init_variable_map()

    def _init_variable_map(self):
        buffer = list()
        position = 0
        for layer in self.structure[1:]:
            prev_node, next_node = layer.weight.shape
            weight_end = position + prev_node * next_node
            bias_end = weight_end + int(layer.bias_flag) * next_node
            buffer.append((position, weight_end, bias_end))
            position = bias_end

        return tuple(buffer)

    def _apply_gradient(self, gradient_vector: np.ndarray):
        for i in range(1, len(self.structure)):
            position, weight_end, bias_end = self.variable_map[i - 1]
            self.structure[i].weight = gradient_vector[position:weight_end].reshape(self.structure[i].weight.shape)
            if self.structure[i].bias_flag:
                self.structure[i].bias = gradient_vector[weight_end:bias_end].reshape(self.structure[i].bias.shape)

    def feedforward(self, batch_data: np.ndarray):
        data = batch_data.copy()

        non_activated = list()
        activated = list()

        for current_layer in self.structure[1:]:
            non_activated_data = np.dot(data, current_layer.weight) + current_layer.bias
            non_activated.append(non_activated_data)

            data = current_layer.activation.activation(non_activated_data)
            activated.append(data)

        return tuple(non_activated), tuple(activated)

    def predict(self, batch_data: np.ndarray):
        data = batch_data.copy()
        for current_layer in self.structure[1:]:
            data = current_layer.activation.activation(np.dot(data, current_layer.weight) + current_layer.bias)

        return data

    def learn(self, batch_data: np.ndarray, batch_target: np.ndarray):
        non_activated, activated = self.feedforward(batch_data)

        position, weight_end, bias_end = self.variable_map[-1]
        gradient = np.zeros(bias_end)

        non_activated_data = non_activated[-1]
        prev_activated_data, activated_data = activated[-2], activated[-1]

        delta = self.loss.derivative(
            activated_data, batch_target
        ) * self.structure[-1].activation.derivative(non_activated_data)

        d_bias = np.mean(delta, axis=0)
        d_weight = np.dot(prev_activated_data.T, delta).reshape((weight_end - position))

        gradient[position:weight_end] = d_weight
        if self.structure[-1].bias_flag:
            gradient[weight_end:bias_end] = d_bias

        for i in range(2, len(self.structure) - 1):
            next_position, next_weight_end, _ = self.variable_map[-(i - 1)]
            position, weight_end, bias_end = self.variable_map[-i]

            next_weight = self.structure[-(i - 1)].weight

            non_activated_data = non_activated[-i]
            prev_activated_data = activated[-(i + 1)]

            delta = np.dot(delta, next_weight.T) * self.structure[-i].activation.derivative(non_activated_data)

            d_bias = np.mean(delta, axis=0)
            d_weight = np.dot(prev_activated_data.T, delta).reshape((weight_end - position))

            gradient[position:weight_end] = d_weight
            if self.structure[-i].bias_flag:
                gradient[weight_end:bias_end] = d_bias

        optimized_gradient = self.optimizer.optimize(gradient_vector=gradient)
        self._apply_gradient(gradient_vector=optimized_gradient)
