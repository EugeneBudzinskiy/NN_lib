import numpy as np

from nnlibrary.nn.AbstractNeuralNetwork import AbstractNeuralNetwork
from nnlibrary.nn.VariableStorage import VariableStorage
from nnlibrary.optimizers import Optimizer
from nnlibrary.losses import Loss


class NeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, structure: tuple, loss: Loss, optimizer: Optimizer):
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer

        self.storage = VariableStorage(structure)

    def feedforward(self, batch_data: np.ndarray):
        data = batch_data.copy()

        non_activated = list()
        activated = list()

        for i in range(len(self.structure) - 1):
            position, weight_end, bias_end = self.storage.map[i]
            weight = self.storage.variables[position:weight_end].reshape(self.storage.shape_map[i])
            bias = self.storage.variables[weight_end:bias_end] if self.structure[i + 1].bias_flag else 0

            non_activated_data = np.dot(data, weight) + bias
            non_activated.append(non_activated_data)

            data = self.structure[i + 1].activation.activate(non_activated_data)
            activated.append(data)

        return tuple(non_activated), tuple(activated)

    def predict(self, batch_data: np.ndarray):
        data = batch_data.copy()

        for i in range(len(self.structure) - 1):
            position, weight_end, bias_end = self.storage.map[i]
            weight = self.storage.variables[position:weight_end].reshape(self.storage.shape_map[i])
            bias = self.storage.variables[weight_end:bias_end] if self.structure[i + 1].bias_flag else 0
            data = self.structure[i + 1].activation.activate(np.dot(data, weight) + bias)

        return data

    def learn(self, batch_data: np.ndarray, batch_target: np.ndarray):
        scale_factor = 1 / len(batch_data)
        non_activated, activated = self.feedforward(batch_data)

        position, weight_end, bias_end = self.storage.map[-1]
        gradient = np.zeros(bias_end)

        non_activated_data = non_activated[-1]
        prev_activated_data, activated_data = activated[-2], activated[-1]

        print(self.loss.loss(batch_target, activated_data))

        delta = self.loss.derivative(activated_data, batch_target) * \
            self.structure[-1].activation.derivative(non_activated_data)

        d_bias = np.mean(delta, axis=0)
        d_weight = scale_factor * np.dot(prev_activated_data.T, delta).reshape((weight_end - position))

        gradient[position:weight_end] = d_weight
        if self.structure[-1].bias_flag:
            gradient[weight_end:bias_end] = d_bias

        for i in range(2, len(self.structure) - 1):
            next_position, next_weight_end, _ = self.storage.map[-(i - 1)]
            position, weight_end, bias_end = self.storage.map[-i]

            next_weight = \
                self.storage.variables[next_position:next_weight_end].reshape(self.storage.shape_map[-(i - 1)])

            non_activated_data = non_activated[-i]
            prev_activated_data = activated[-(i + 1)]

            delta = np.dot(delta, next_weight.T) * self.structure[-i].activation.derivative(non_activated_data)

            d_bias = np.mean(delta, axis=0)
            d_weight = scale_factor * np.dot(prev_activated_data.T, delta).reshape((weight_end - position))

            gradient[position:weight_end] = d_weight
            if self.structure[-i].bias_flag:
                gradient[weight_end:bias_end] = d_bias

        self.optimizer.optimize(trainable_variables=self.storage.variables, gradient_vector=gradient)
