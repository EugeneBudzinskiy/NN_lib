import numpy as np

from typing import Callable

# TODO Refactor this structure of Layer


class AbstractLayer:
    def __init__(self, node_count: int):
        self._node_count = node_count

    @property
    def node_count(self) -> int:
        return self._node_count


class AbstractActivationLayer(AbstractLayer):
    def __init__(self, node_count: int, activation, bias_flag: bool = True, trainable: bool = True):
        super(AbstractActivationLayer, self).__init__(node_count=node_count)
        self._activation = activation
        self._bias_flag = bias_flag
        self._trainable = trainable

    @property
    def activation(self) -> Callable:
        return self._activation

    @property
    def bias_flag(self) -> bool:
        return self._bias_flag

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        self._trainable = value
