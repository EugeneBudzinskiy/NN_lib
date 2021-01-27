from functions.Activations import AbstractActivation
from layers.LayerTypes import ActivationLayer
from layers.LayerTypes import InputLayer


class Layers:
    def __init__(self):
        self.Input = Input
        self.Dense = Dense


class Layer:
    def __init__(self, node_count: int):
        self.node_count = node_count

    def get_node_count(self):
        return self.node_count


class Input(InputLayer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class Dense(ActivationLayer):
    def __init__(self, node_count: int, activation: AbstractActivation):
        super(Dense, self).__init__(node_count, activation)
