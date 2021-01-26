from SingletonMeta import SingletonMeta
from Activations import Activations
from Losses import Losses
from Optimizers import Optimizers


class Functions(metaclass=SingletonMeta):
    def __init__(self):
        self.activations = Activations()
        self.losses = Losses()
        self.optimizers = Optimizers()
