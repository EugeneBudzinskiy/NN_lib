import numpy as np

from nnlibrary.activations import Sigmoid
from nnlibrary.activations import Softmax
from nnlibrary.differentiators import Derivative
from nnlibrary.differentiators import Gradient
from nnlibrary.layer_structures import AbstractLayerStructure
from nnlibrary.layer_structures import LayerStructure
from nnlibrary.layers import AbstractActivationLayer
from nnlibrary.layers import AbstractLayer
from nnlibrary.losses import AbstractLoss
from nnlibrary.losses import MeanSquaredError
from nnlibrary.losses import CategoricalCrossentropy
from nnlibrary.models import AbstractModel
from nnlibrary.optimizers import AbstractOptimizer
from nnlibrary.optimizers import SGD
from nnlibrary.reductions import ReductionNone
from nnlibrary.variables import AbstractInitializer
from nnlibrary.variables import TrainableVariables


class Sequential(AbstractModel):
    def __init__(self):
        self.layer_structure = LayerStructure()

        # noinspection PyTypeChecker
        self.core: SequentialCompiledCore = None

    @property
    def is_compiled(self) -> bool:
        return isinstance(self.core, SequentialCompiledCore)

    def get_variables(self) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.trainable_variables.get_all()

    @property
    def loss(self) -> AbstractLoss:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.loss

    @property
    def optimizer(self) -> AbstractOptimizer:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.optimizer

    def add(self, layer: AbstractLayer):
        if self.is_compiled:
            raise Exception()  # TODO Custom Exception (not changeable after compile)

        self.layer_structure.add_layer(layer=layer)

    def compile(self,
                optimizer: AbstractOptimizer = None,
                loss: AbstractLoss = None,
                weight_initializer: AbstractInitializer = None,
                bias_initializer: AbstractInitializer = None):

        if self.is_compiled:
            raise Exception()  # TODO Custom Exception (already compiled)

        optimizer = SGD() if optimizer is None else optimizer
        loss = MeanSquaredError() if loss is None else loss

        self.core = SequentialCompiledCore(
            layer_structure=self.layer_structure,
            optimizer=optimizer,
            loss=loss,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )

    def feedforward(self, x: np.ndarray) -> (np.ndarray, [np.ndarray], [np.ndarray]):
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        a = x.copy() if x.ndim > 1 else x.reshape(1, -1)
        z_list, a_list = list(), list()
        a_list.append(a)

        for i in range(1, self.layer_structure.layers_number):
            current_layer = self.layer_structure.get_layer(layer_number=i)
            current_weight, current_bias = self.core.unpack_variables(layer_number=i)

            z = np.dot(a, current_weight) + current_bias
            z_list.append(z)

            if not isinstance(current_layer, AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            a = current_layer.activation(x=z)
            a_list.append(a)

        return a_list.pop(), z_list, a_list

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        output, _, _ = self.feedforward(x=x)
        return output

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        x = x if x.ndim > 1 else x.reshape(1, -1)
        y = y if y.ndim > 1 else y.reshape(1, -1)

        output, z_list, a_list = self.feedforward(x=x)
        layers_number = self.layer_structure.layers_number

        loss_gradient = self.core.loss_gradient(y_target=y, y_predicted=output)
        delta = loss_gradient * self.core.activation_derivatives[-1](x=z_list[-1])

        d_weight = np.dot(a_list[-1].T, delta)
        d_bias = np.sum(delta, axis=0).reshape(1, -1)

        gradient_list = list()
        gradient_list.append(d_bias)
        gradient_list.append(d_weight)

        for i in range(1, layers_number - 1):
            j = layers_number - i - 1
            previous_weight, _ = self.core.unpack_variables(layer_number=j + 1)

            next_delta = np.dot(delta, previous_weight.T)
            delta = next_delta * self.core.activation_derivatives[j](x=z_list[j - 1])

            d_weight = np.dot(a_list[j - 1].T, delta)
            d_bias = np.sum(delta, axis=0).reshape(1, -1)

            gradient_list.append(d_bias)
            gradient_list.append(d_weight)

        gradient_list.reverse()
        gradient_vector = np.concatenate(gradient_list, axis=None) / y.shape[0]
        return gradient_vector

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            shuffle: bool = True):

        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        x = x if x.ndim > 1 else x.reshape(1, -1)
        y = y if y.ndim > 1 else y.reshape(1, -1)

        size = x.shape[0]
        indexes = np.arange(size)
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indexes)

            for i in range(0, size, batch_size):
                idx = indexes[i:i + batch_size]
                gradient_vector = self.backpropagation(x=x[idx], y=y[idx])
                adjustment = self.core.optimizer(gradient_vector=gradient_vector)

                self.core.trainable_variables.set_all(
                    value=self.core.trainable_variables.get_all() + adjustment
                )


class SequentialCompiledCore:
    def __init__(self,
                 layer_structure: AbstractLayerStructure,
                 optimizer: AbstractOptimizer,
                 loss: AbstractLoss,
                 weight_initializer: AbstractInitializer,
                 bias_initializer: AbstractInitializer):

        self.derivative = Derivative()
        self.gradient = Gradient()

        self.trainable_variables = TrainableVariables()
        self.layer_structure = layer_structure

        self.optimizer = optimizer
        self.loss = loss

        self.trainable_variables.init_variables(
            layer_structure=self.layer_structure,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )

        self.loss_gradient = self.get_loss_gradient()
        self.activation_derivatives = self.get_activation_derivatives()

    def get_loss_gradient(self) -> callable:
        def loss_wrapper(y_target: np.ndarray):
            return lambda x: self.loss(y_predicted=x, y_target=y_target, reduction=ReductionNone())

        if isinstance(self.loss, CategoricalCrossentropy):
            last_layer = self.layer_structure.get_layer(layer_number=self.layer_structure.layers_number - 1)
            softmax = Softmax()

            if not isinstance(last_layer, AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            if isinstance(last_layer.activation, Softmax) or \
                    isinstance(last_layer.activation, Sigmoid):
                raise Exception('empty')

            else:
                def shortcut(y_target, y_predicted):
                    return - 1. / y_predicted

                return shortcut

        else:
            return lambda y_target, y_predicted: self.gradient(
                func=loss_wrapper(y_target=y_target), x=y_predicted
            )

    def get_activation_derivatives(self) -> [callable]:
        derivatives = list([None])
        for i in range(1, self.layer_structure.layers_number):
            if None:
                raise Exception('empty')

            else:
                current_layer = self.layer_structure.get_layer(layer_number=i)

                if not isinstance(current_layer, AbstractActivationLayer):
                    raise Exception()  # TODO Custom Exception

                activation = current_layer.activation
                derivatives.append(lambda x: self.derivative(func=activation, x=x))

        return derivatives

    def unpack_variables(self, layer_number: int) -> (np.ndarray, np.ndarray):
        current_layer = self.layer_structure.get_layer(layer_number=layer_number)
        current_vars = self.trainable_variables.get_single(layer_number=layer_number)

        current_node_count = current_layer.node_count
        previous_node_count = len(current_vars) // (current_node_count + 1)
        w_size = previous_node_count * current_node_count

        current_weight = current_vars[:w_size].reshape((previous_node_count, current_node_count))
        current_bias = current_vars[w_size:].reshape((1, -1))

        return current_weight, current_bias


# import time
# from nnlibrary import errors
# from nnlibrary import differentiators
# from nnlibrary import layers

# class Sequential_:
#     def __init__(self):
#         self.is_compiled = False
#         self.diff = differentiators.Gradient()
#
#         self.input_layer = None
#         self.layers = list()
#
#         self.variables = None
#         self.variables_map = tuple()
#
#         self.variable_count = 0
#         self.layer_count = 0
#
#         self.optimizer = None
#         self.loss = None
#
#     def add(self, layer):
#         if self.is_compiled:
#             raise errors.TryModifyCompiledNN
#
#         if not isinstance(layer, layers.AbstractLayer):
#             raise errors.IsNotALayer(layer)
#
#         if isinstance(layer, layers.AbstractActivationLayer):
#             self.layers.append(layer)
#         else:
#             if self.input_layer is None:
#                 self.input_layer = layer
#             else:
#                 raise errors.InputLayerAlreadyDefined
#
#     def pop(self):
#         if self.is_compiled:
#             raise errors.TryModifyCompiledNN
#
#         if len(self.layers) == 0:
#             raise errors.NothingToPop
#
#         self.layers.pop()
#
#     def get_layer(self, index: int = None):
#         if not self.is_compiled:
#             raise errors.NotCompiled
#
#         if index is None:
#             raise errors.ProvideLayerIndex
#
#         if len(self.layers) <= index:
#             raise errors.WrongLayerIndex
#
#         return self.layers[index]
#
#     @staticmethod
#     def _get_init_weight(prev_node_count: int, next_node_count: int):
#         coefficient = np.sqrt(1 / next_node_count)
#         return coefficient * np.random.randn(prev_node_count * next_node_count)
#
#     @staticmethod
#     def _var_map_packer(a_size: int, b_size: int, position: int):
#         w_end = position + a_size * b_size
#         b_end = w_end + b_size
#         return b_end, (position, w_end, b_end, (a_size, b_size))
#
#     def weight_initialization(self):
#         a_size = self.input_layer.node_count
#         b_size = self.layers[0].node_count
#
#         position = 0
#         position, package = self._var_map_packer(a_size, b_size, position)
#         variables_map = [package]
#         self.variables = np.concatenate((self._get_init_weight(a_size, b_size), np.zeros(b_size)), axis=None)
#
#         for i in range(1, self.layer_count):
#             a_size = self.layers[i - 1].node_count
#             b_size = self.layers[i].node_count
#
#             position, package = self._var_map_packer(a_size, b_size, position)
#             variables_map.append(package)
#
#             current_weight = self._get_init_weight(a_size, b_size)
#             self.variables = np.concatenate((self.variables, current_weight, np.zeros(b_size)), axis=None)
#
#         self.variables_map = tuple(variables_map)
#         self.variable_count = len(self.variables)
#
#     def compile(self, optimizer=None, loss=None):
#         if self.is_compiled:
#             raise errors.AlreadyCompiled
#
#         if len(self.layers) == 0:
#             raise errors.WrongStructure
#
#         if optimizer is None:
#             raise errors.OptimizerNotSpecify
#
#         if loss is None:
#             raise errors.LossNotSpecify
#
#         if self.input_layer is None:
#             raise errors.InputLayerNotDefined
#
#         self.layer_count = len(self.layers)
#
#         self.loss = loss
#         self.optimizer = optimizer
#
#         self.weight_initialization()
#         self.is_compiled = True
#
#     def get_weight(self, layer_number):
#         if not self.is_compiled:
#             raise errors.NotCompiled
#
#         w_start, w_end, _, w_shape = self.variables_map[layer_number]
#         return self.variables[w_start:w_end].reshape(w_shape)
#
#     def get_bias(self, layer_number):
#         if not self.is_compiled:
#             raise errors.NotCompiled
#
#         _, b_start, b_end, _ = self.variables_map[layer_number]
#         return self.variables[b_start:b_end]
#
#     def get_activation(self, layer_number):
#         return self.layers[layer_number].activation
#
#     def get_bias_flag(self, layer_number):
#         return self.layers[layer_number].bias_flag
#
#     def predict(self, x: np.ndarray):
#         data = x.copy()
#         for i in range(self.layer_count):
#             data = self.get_activation(i)(np.dot(data, self.get_weight(i)) + self.get_bias(i))
#         return data
#
#     def feedforward(self, x: np.ndarray):
#         a = x.copy()
#         if a.ndim == 1:
#             a = a.reshape((1, -1))
#
#         z_list = []
#         a_list = [a]
#
#         for i in range(self.layer_count):
#             z = np.dot(a, self.get_weight(layer_number=i)) + self.get_bias(i)
#             z_list.append(z)
#
#             a = self.get_activation(layer_number=i)(z)
#             a_list.append(a)
#
#         return a_list.pop(), z_list, a_list
#
#     def loss_wrapper(self, target):
#         return lambda x: self.loss(y_predicted=x, y_target=target)
#
#     def back_propagation(self, x, y, batch_size):
#         output, z_list, a_list = self.feedforward(x=x)
#         loss = self.loss_wrapper(target=y)
#
#         delta = self.diff(loss, output) * self.diff(self.get_activation(-1), z_list[-1])
#         d_weight = np.dot(a_list[-1].T, delta)
#         d_bias = np.sum(delta, axis=0) if self.get_bias_flag(-1) else np.zeros(self.layers[-1].node_count)
#         gradient = np.concatenate((d_weight, d_bias), axis=None)
#
#         for i in range(self.layer_count - 2, -1, -1):
#             delta = np.dot(delta, self.get_weight(i + 1).T) * self.diff(self.get_activation(i), z_list[i])
#             d_weight = np.dot(a_list[i].T, delta)
#             d_bias = np.sum(delta, axis=0) if self.get_bias_flag(i) else np.zeros(self.layers[i].node_count)
#             gradient = np.concatenate((d_weight, d_bias, gradient), axis=None)
#
#         gradient /= batch_size
#         self.optimizer(trainable_variables=self.variables, gradient_vector=gradient)
#
#     def fit(self,
#             x: np.ndarray,
#             y: np.ndarray,
#             epochs: int = 1,
#             batch_size: int = None,
#             shuffle: bool = True):
#
#         if not self.is_compiled:
#             raise errors.NotCompiled
#
#         if batch_size is None:
#             batch_size = 32
#
#         if shuffle:
#             self.random_fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
#         else:
#             self.static_fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
#
#     @staticmethod
#     def fit_progress_bar(i, total, epoch, epochs, time_start):
#         # prefix = 'Progress' if epochs == 1 else f'Epoch {epoch + 1}/{epochs} ||| Progress:'
#         # progress_bars(iteration=i + 1, time_passed=time.time() - time_start, total=total, prefix=prefix)
#         pass
#
#     def static_fit(self,
#                    x: np.ndarray,
#                    y: np.ndarray,
#                    epochs: int = 1,
#                    batch_size: int = None):
#
#         x_train = x.copy()
#         y_train = y.copy()
#
#         total = len(y_train) // batch_size
#
#         for epoch in range(epochs):
#             time_start = time.time()
#             for i in range(total):
#                 d_start, d_end = i * batch_size, (i + 1) * batch_size
#                 self.back_propagation(x=x_train[d_start:d_end], y=y_train[d_start:d_end], batch_size=batch_size)
#                 self.fit_progress_bar(i=i, total=total, epoch=epoch, epochs=epochs, time_start=time_start)
#
#     def random_fit(self,
#                    x: np.ndarray,
#                    y: np.ndarray,
#                    epochs: int = 1,
#                    batch_size: int = None):
#
#         x_train = x.copy()
#         y_train = y.copy()
#
#         ln = len(y_train)
#         total = ln // batch_size
#
#         for epoch in range(epochs):
#             time_start = time.time()
#             for i in range(total):
#                 indexes = np.random.randint(low=0, high=ln, size=batch_size)
#                 self.back_propagation(x=x_train[indexes], y=y_train[indexes], batch_size=batch_size)
#                 self.fit_progress_bar(i=i, total=total, epoch=epoch, epochs=epochs, time_start=time_start)
#
#     def load_weights(self, file_name: str = "weights"):
#         self.variables = np.loadtxt(file_name + '.csv', delimiter=',')
#
#     def save_weights(self, file_name: str = "weights"):
#         np.savetxt(file_name + '.csv', self.variables, delimiter=',')
#
#     def save_model(self):
#         pass
#
#     def load_model(self):
#         pass
#
#     def summary(self):
#         pass
