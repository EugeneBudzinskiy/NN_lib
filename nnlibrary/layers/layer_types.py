from layer_tools import Layer
from layer_tools import ActivationLayer


class Input(Layer):
    def __init__(self, node_count: int):
        super(Input, self).__init__(node_count=node_count)


class Dense(ActivationLayer):
    def __init__(self,
                 node_count: int,
                 activation,
                 bias_flag: bool = True,
                 trainable: bool = True):

        super(Dense, self).__init__(
            node_count=node_count,
            activation=activation,
            bias_flag=bias_flag,
            trainable=trainable
        )
