from nnlibrary.activation import AbstractActivation
from nnlibrary.layers.LayerTypes import ActivationLayer
from nnlibrary.layers.LayerTypes import InputLayer


class Layers:
    def __init__(self):
        self.Input = Input
        self.Dense = Dense


class Input(InputLayer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class Dense(ActivationLayer):
    def __init__(self, node_count: int, activation: AbstractActivation):
        super(Dense, self).__init__(node_count, activation)
