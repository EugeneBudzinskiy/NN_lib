import numpy as np

from nnlibrary.activation import AbstractActivation


class Layer:
    def __init__(self, node_count: int):
        self._node_count = node_count

    @property
    def node_count(self):
        return self._node_count


class InputLayer(Layer):
    def __init__(self, node_count: int):
        super(InputLayer, self).__init__(node_count=node_count)


class ActivationLayer(Layer):
    def __init__(self, node_count: int, activation: AbstractActivation, bias_flag: bool = True):
        super(ActivationLayer, self).__init__(node_count=node_count)
        self._activation = activation

        self._bias_flag = bias_flag
        self._position = None
        self._weight = None
        self._bias = None

    @property
    def activation(self):
        return self._activation

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, vector):
        self._weight += vector

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, vector):
        self._bias += vector

    @property
    def bias_flag(self):
        return self._bias_flag

    def layer_initialization(self, previous_layer: Layer):
        previous_node_count = previous_layer.node_count
        coefficient = np.sqrt(1 / (previous_node_count + self.node_count))

        self._weight = np.random.rand(previous_node_count * self.node_count).reshape(
            (previous_node_count, self.node_count)) * coefficient

        self._bias = np.zeros(self.node_count) if self._bias_flag else 0

