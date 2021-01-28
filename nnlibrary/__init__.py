from nnlibrary.activation import Activations
from nnlibrary.constructor import Constructor
from nnlibrary.layers import Layers
from nnlibrary.losses import Losses
from nnlibrary.optimizers import Optimizers
from nnlibrary.singleton import SingletonMeta


class NNLib(metaclass=SingletonMeta):
    def __init__(self):
        self.activation = Activations()
        self.layers = Layers()
        self.losses = Losses()
        self.optimizers = Optimizers()
        self.constructor = Constructor()
