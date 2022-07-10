import time

import numpy as np

from nnlibrary import errors
from nnlibrary import differentiators
from nnlibrary import layers

from nnlibrary.differentiators import SimpleDifferentiator
from nnlibrary.models import AbstractModel
from nnlibrary.layers import AbstractLayer
from nnlibrary.layers import AbstractActivationLayer
from nnlibrary.layer_structures import LayerStructure
from nnlibrary.optimizers import AbstractOptimizer
from nnlibrary.losses import AbstractLoss
from nnlibrary.variables import TrainableVariables


class Sequential(AbstractModel):
    def __init__(self):
        self.is_compiled = False

        self.diff = SimpleDifferentiator()
        self.layer_structure = LayerStructure()
        self.trainable_variables = TrainableVariables()

        self.optimizer = None
        self.loss = None

    def add(self, layer: AbstractLayer):
        self.layer_structure.add_layer(layer=layer)

    def compile(self,
                optimizer: AbstractOptimizer = None,
                loss: AbstractLoss = None):

        self.optimizer = optimizer
        self.loss = loss

        self.trainable_variables.init_variables(layer_structure=self.layer_structure)
        self.is_compiled = True

    def __unpack_weight_and_bias(self, layer_number: int) -> (np.ndarray, np.ndarray):
        current_layer = self.layer_structure.get_layer(layer_number=layer_number)
        current_vars = self.trainable_variables.get_single(layer_number=layer_number)

        current_node_count = current_layer.node_count
        previous_node_count = len(current_vars) // (current_node_count + 1)
        w_size = previous_node_count * current_node_count

        current_weight = current_vars[:w_size].reshape((current_node_count, previous_node_count))
        current_bias = current_vars[w_size:]

        return current_weight, current_bias

    def feedforward(self, x: np.ndarray) -> (np.ndarray, [np.ndarray], [np.ndarray]):
        a = x.copy().flatten()
        z_list, a_list = list(), list()
        a_list.append(a)

        for i in range(1, self.layer_structure.layers_number):
            current_layer = self.layer_structure.get_layer(layer_number=i)
            current_weight, current_bias = self.__unpack_weight_and_bias(layer_number=i)

            z = np.dot(current_weight, a) + current_bias
            z_list.append(z)

            if not isinstance(current_layer, AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            a = current_layer.activation(x=z)
            a_list.append(a)

        return a_list.pop(), z_list, a_list

    def predict(self, x: np.ndarray) -> np.ndarray:
        output, _, _ = self.feedforward(x=x)
        return output

    @staticmethod
    def loss_wrapper(loss: AbstractLoss, target: np.ndarray) -> callable:
        return lambda x: loss(y_predicted=x, y_target=target)

    def backpropagation(self, x: np.ndarray, y: np.ndarray, batch_size: int):
        if not isinstance(self.loss, AbstractLoss):
            raise Exception()  # TODO Custom Exception

        output, z_list, a_list = self.feedforward(x=x)
        layers_number = self.layer_structure.layers_number
        current_layer = self.layer_structure.get_layer(layer_number=layers_number - 1)

        if not isinstance(current_layer, AbstractActivationLayer):
            raise Exception()  # TODO Custom Exception

        loss_fixed = self.loss_wrapper(loss=self.loss, target=y)

        delta = self.diff(func=loss_fixed, x=output) * \
            self.diff(func=current_layer.activation, x=z_list[-1])
        d_weight = np.dot(a_list[-1].T, delta)
        d_bias = np.sum(delta, axis=0)

        gradient_list = [(d_weight, d_bias)]

        for i in range(layers_number - 1):
            j = layers_number - i - 2

            current_layer = self.layer_structure.get_layer(layer_number=j)
            previous_weight, _ = self.__unpack_weight_and_bias(layer_number=j + 1)

            if not isinstance(current_layer, AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            delta = np.dot(delta, previous_weight.T) * self.diff(func=current_layer.activation, x=z_list[j])
            d_weight = np.dot(a_list[j].T, delta)
            d_bias = np.sum(delta, axis=0)
            gradient_list.append((d_weight, d_bias))

        gradient_list.reverse()
        print(gradient_list)  # TODO Probably debug is needed

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epoch_number: int = 1,
            batch_size: int = 32,
            shuffle: bool = False):
        pass


class Sequential_:
    def __init__(self):
        self.is_compiled = False
        self.diff = differentiators.SimpleDifferentiator()

        self.input_layer = None
        self.layers = list()

        self.variables = None
        self.variables_map = tuple()

        self.variable_count = 0
        self.layer_count = 0

        self.optimizer = None
        self.loss = None

    def add(self, layer):
        if self.is_compiled:
            raise errors.TryModifyCompiledNN

        if not isinstance(layer, layers.AbstractLayer):
            raise errors.IsNotALayer(layer)

        if isinstance(layer, layers.AbstractActivationLayer):
            self.layers.append(layer)
        else:
            if self.input_layer is None:
                self.input_layer = layer
            else:
                raise errors.InputLayerAlreadyDefined

    def pop(self):
        if self.is_compiled:
            raise errors.TryModifyCompiledNN

        if len(self.layers) == 0:
            raise errors.NothingToPop

        self.layers.pop()

    def get_layer(self, index: int = None):
        if not self.is_compiled:
            raise errors.NotCompiled

        if index is None:
            raise errors.ProvideLayerIndex

        if len(self.layers) <= index:
            raise errors.WrongLayerIndex

        return self.layers[index]

    @staticmethod
    def _get_init_weight(prev_node_count: int, next_node_count: int):
        coefficient = np.sqrt(1 / next_node_count)
        return coefficient * np.random.randn(prev_node_count * next_node_count)

    @staticmethod
    def _var_map_packer(a_size: int, b_size: int, position: int):
        w_end = position + a_size * b_size
        b_end = w_end + b_size
        return b_end, (position, w_end, b_end, (a_size, b_size))

    def weight_initialization(self):
        a_size = self.input_layer.node_count
        b_size = self.layers[0].node_count

        position = 0
        position, package = self._var_map_packer(a_size, b_size, position)
        variables_map = [package]
        self.variables = np.concatenate((self._get_init_weight(a_size, b_size), np.zeros(b_size)), axis=None)

        for i in range(1, self.layer_count):
            a_size = self.layers[i - 1].node_count
            b_size = self.layers[i].node_count

            position, package = self._var_map_packer(a_size, b_size, position)
            variables_map.append(package)

            current_weight = self._get_init_weight(a_size, b_size)
            self.variables = np.concatenate((self.variables, current_weight, np.zeros(b_size)), axis=None)

        self.variables_map = tuple(variables_map)
        self.variable_count = len(self.variables)

    def compile(self, optimizer=None, loss=None):
        if self.is_compiled:
            raise errors.AlreadyCompiled

        if len(self.layers) == 0:
            raise errors.WrongStructure

        if optimizer is None:
            raise errors.OptimizerNotSpecify

        if loss is None:
            raise errors.LossNotSpecify

        if self.input_layer is None:
            raise errors.InputLayerNotDefined

        self.layer_count = len(self.layers)

        self.loss = loss
        self.optimizer = optimizer

        self.weight_initialization()
        self.is_compiled = True

    def get_weight(self, layer_number):
        if not self.is_compiled:
            raise errors.NotCompiled

        w_start, w_end, _, w_shape = self.variables_map[layer_number]
        return self.variables[w_start:w_end].reshape(w_shape)

    def get_bias(self, layer_number):
        if not self.is_compiled:
            raise errors.NotCompiled

        _, b_start, b_end, _ = self.variables_map[layer_number]
        return self.variables[b_start:b_end]

    def get_activation(self, layer_number):
        return self.layers[layer_number].activation

    def get_bias_flag(self, layer_number):
        return self.layers[layer_number].bias_flag

    def predict(self, x: np.ndarray):
        data = x.copy()
        for i in range(self.layer_count):
            data = self.get_activation(i)(np.dot(data, self.get_weight(i)) + self.get_bias(i))
        return data

    def feedforward(self, x: np.ndarray):
        a = x.copy()
        if a.ndim == 1:
            a = a.reshape((1, -1))

        z_list = []
        a_list = [a]

        for i in range(self.layer_count):
            z = np.dot(a, self.get_weight(layer_number=i)) + self.get_bias(i)
            z_list.append(z)

            a = self.get_activation(layer_number=i)(z)
            a_list.append(a)

        return a_list.pop(), z_list, a_list

    def loss_wrapper(self, target):
        return lambda x: self.loss(y_predicted=x, y_target=target)

    def back_propagation(self, x, y, batch_size):
        output, z_list, a_list = self.feedforward(x=x)
        loss = self.loss_wrapper(target=y)

        delta = self.diff(loss, output) * self.diff(self.get_activation(-1), z_list[-1])
        d_weight = np.dot(a_list[-1].T, delta)
        d_bias = np.sum(delta, axis=0) if self.get_bias_flag(-1) else np.zeros(self.layers[-1].node_count)
        gradient = np.concatenate((d_weight, d_bias), axis=None)

        for i in range(self.layer_count - 2, -1, -1):
            delta = np.dot(delta, self.get_weight(i + 1).T) * self.diff(self.get_activation(i), z_list[i])
            d_weight = np.dot(a_list[i].T, delta)
            d_bias = np.sum(delta, axis=0) if self.get_bias_flag(i) else np.zeros(self.layers[i].node_count)
            gradient = np.concatenate((d_weight, d_bias, gradient), axis=None)

        gradient /= batch_size
        self.optimizer(trainable_variables=self.variables, gradient_vector=gradient)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = None,
            shuffle: bool = True):

        if not self.is_compiled:
            raise errors.NotCompiled

        if batch_size is None:
            batch_size = 32

        if shuffle:
            self.random_fit(x=x, y=y, epochs=epochs, batch_size=batch_size)
        else:
            self.static_fit(x=x, y=y, epochs=epochs, batch_size=batch_size)

    @staticmethod
    def fit_progress_bar(i, total, epoch, epochs, time_start):
        # prefix = 'Progress' if epochs == 1 else f'Epoch {epoch + 1}/{epochs} ||| Progress:'
        # progress_bars(iteration=i + 1, time_passed=time.time() - time_start, total=total, prefix=prefix)
        pass

    def static_fit(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   epochs: int = 1,
                   batch_size: int = None):

        x_train = x.copy()
        y_train = y.copy()

        total = len(y_train) // batch_size

        for epoch in range(epochs):
            time_start = time.time()
            for i in range(total):
                d_start, d_end = i * batch_size, (i + 1) * batch_size
                self.back_propagation(x=x_train[d_start:d_end], y=y_train[d_start:d_end], batch_size=batch_size)
                self.fit_progress_bar(i=i, total=total, epoch=epoch, epochs=epochs, time_start=time_start)

    def random_fit(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   epochs: int = 1,
                   batch_size: int = None):

        x_train = x.copy()
        y_train = y.copy()

        ln = len(y_train)
        total = ln // batch_size

        for epoch in range(epochs):
            time_start = time.time()
            for i in range(total):
                indexes = np.random.randint(low=0, high=ln, size=batch_size)
                self.back_propagation(x=x_train[indexes], y=y_train[indexes], batch_size=batch_size)
                self.fit_progress_bar(i=i, total=total, epoch=epoch, epochs=epochs, time_start=time_start)

    def load_weights(self, file_name: str = "weights"):
        self.variables = np.loadtxt(file_name + '.csv', delimiter=',')

    def save_weights(self, file_name: str = "weights"):
        np.savetxt(file_name + '.csv', self.variables, delimiter=',')

    def save_model(self):
        pass

    def load_model(self):
        pass

    def summary(self):
        pass
