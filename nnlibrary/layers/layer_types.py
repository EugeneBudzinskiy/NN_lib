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
    def __init__(self, node_count: int, activation, bias_flag: bool = True, trainable: bool = True):
        super(ActivationLayer, self).__init__(node_count=node_count)
        self._activation = activation
        self._bias_flag = bias_flag
        self.trainable = trainable

    @property
    def activation(self):
        return self._activation

    @property
    def bias_flag(self):
        return self._bias_flag


class Input(InputLayer):
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
