# import numpy as np
#
# from nnlibrary.losses import Loss
# from nnlibrary.models.AbstractNeuralNetwork import AbstractNeuralNetwork
# from nnlibrary.models.VariableStorage import VariableStorage
# from nnlibrary.optimizers import Optimizer


class Model:
    def __init__(self):
        pass

    def compile(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


# class NeuralNetwork(AbstractNeuralNetwork):
#     def __init__(self, structure: tuple, loss: Loss, optimizer: Optimizer):
#         self.structure = structure
#         self.loss = loss
#         self.optimizer = optimizer
#
#         self.storage = VariableStorage(structure)
#
#     def feedforward(self, batch_data: np.ndarray):
#         data = batch_data.copy()
#
#         non_activated = list()
#         activated = list()
#
#         for i in range(len(self.structure) - 1):
#             position, weight_end, bias_end = self.storage.map[i]
#             weight = self.storage.variables[position:weight_end].reshape(self.storage.shape_map[i])
#             bias = self.storage.variables[weight_end:bias_end] if self.structure[i + 1].bias_flag else 0
#
#             non_activated_data = np.dot(data, weight) + bias
#             non_activated.append(non_activated_data)
#
#             data = self.structure[i + 1].activation.activate(non_activated_data)
#             activated.append(data)
#
#         return tuple(non_activated), tuple(activated)
#
#     def predict(self, batch_data: np.ndarray):
#         data = batch_data.copy()
#
#         for i in range(len(self.structure) - 1):
#             position, weight_end, bias_end = self.storage.map[i]
#             weight = self.storage.variables[position:weight_end].reshape(self.storage.shape_map[i])
#             bias = self.storage.variables[weight_end:bias_end] if self.structure[i + 1].bias_flag else 0
#             data = self.structure[i + 1].activation.activate(np.dot(data, weight) + bias)
#
#         return data
#
#     def learn(self, batch_data: np.ndarray, batch_target: np.ndarray):
#         non_activated, activated = self.feedforward(batch_data)
#
#         position, weight_end, bias_end = self.storage.map[-1]
#         gradient = np.zeros(bias_end)
#
#         non_activated_data = non_activated[-1]
#         prev_activated_data, activated_data = activated[-2], activated[-1]
#
#         delta = \
#             self.loss.derivative(y_predicted=activated_data, y_target=batch_target) * \
#             self.structure[-1].activation.derivative(non_activated_data)
#
#         d_bias = np.sum(delta, axis=0)
#         d_weight = np.dot(prev_activated_data.T, delta).reshape((weight_end - position))
#
#         gradient[position:weight_end] = d_weight
#         if self.structure[-1].bias_flag:
#             gradient[weight_end:bias_end] = d_bias
#
#         for i in range(2, len(self.structure) - 1):
#             next_position, next_weight_end, _ = self.storage.map[-(i - 1)]
#             position, weight_end, bias_end = self.storage.map[-i]
#
#             next_weight = \
#                 self.storage.variables[next_position:next_weight_end].reshape(self.storage.shape_map[-(i - 1)])
#
#             non_activated_data = non_activated[-i]
#             prev_activated_data = activated[-(i + 1)]
#
#             delta = np.dot(delta, next_weight.T) * self.structure[-i].activation.derivative(non_activated_data)
#
#             d_bias = np.sum(delta, axis=0)
#             d_weight = np.dot(prev_activated_data.T, delta).reshape((weight_end - position))
#
#             gradient[position:weight_end] = d_weight
#             if self.structure[-i].bias_flag:
#                 gradient[weight_end:bias_end] = d_bias
#
#         self.optimizer.optimize(trainable_variables=self.storage.variables, gradient_vector=gradient)
#
#     def train(self,
#               data: np.ndarray,
#               target: np.ndarray,
#               epoch: int = 1,
#               batch_size: int = 32,
#               stochastic: bool = True):
#
#         data_len = len(target)
#         pass_number = data_len // batch_size
#
#         for eph in range(epoch):
#             print(f'Epoch {eph + 1}: ')
#             for i in range(pass_number):
#                 if stochastic:
#                     indexes = np.random.choice(data_len, batch_size)
#                 else:
#                     indexes = np.arange(start=i * batch_size, stop=(i + 1) * batch_size)
#
#                 self.learn(batch_data=data[indexes], batch_target=target[indexes])
#
#     def test_accuracy(self, data: np.ndarray, target: np.ndarray, batch_size: int = 1):
#         data_len = len(target)
#         pass_number = data_len // batch_size
#         wrong_counter = 0
#
#         for i in range(pass_number):
#             result = self.predict(data[i * batch_size:(i + 1) * batch_size])
#             current_target = target[i * batch_size:(i + 1) * batch_size]
#
#             wrong_counter += np.count_nonzero(np.argmax(result, axis=1) - np.argmax(current_target, axis=1))
#
#         return 1 - wrong_counter / (pass_number * batch_size)
