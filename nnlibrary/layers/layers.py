from nnlibrary import layers
from nnlibrary.activations import AbstractActivation


class Input(layers.AbstractLayer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class Dense(layers.AbstractActivationLayer):
    def __init__(self,
                 node_count: int,
                 activation: AbstractActivation,
                 bias_flag: bool = True,
                 trainable: bool = True):

        super(Dense, self).__init__(
            node_count=node_count,
            activation=activation,
            bias_flag=bias_flag,
            trainable=trainable
        )
