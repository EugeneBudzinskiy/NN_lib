from functions.Activations import AbstractActivation


class Layers:
    def __init__(self):
        self.Input = Input
        self.Dense = Dense


class Layer:
    def __init__(self, node_count: int):
        self.node_count = node_count

    def get_node_count(self):
        return self.node_count


class InputLayer(Layer):
    def __init__(self, node_count: int):
        super(InputLayer, self).__init__(node_count=node_count)


class Input(InputLayer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class ActivationLayer(Layer):
    def __init__(self, node_count: int, activation: AbstractActivation):
        super(ActivationLayer, self).__init__(node_count=node_count)
        self.activation = activation

    def activation(self):
        return self.activation


class Dense(ActivationLayer):
    def __init__(self, node_count: int, activation: AbstractActivation):
        super(Dense, self).__init__(node_count, activation)
