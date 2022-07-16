from nnlibrary.layers import AbstractLayer
from nnlibrary.layers import AbstractActivationLayer
from nnlibrary.activations import AbstractActivation


class Input(AbstractLayer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class Dense(AbstractActivationLayer):
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
