from singleton import SingletonMeta
from functions.Activations import Activations
from functions.Losses import Losses
from functions.Optimizers import Optimizers


class Functions(metaclass=SingletonMeta):
    def __init__(self):
        self.activations = Activations()
        self.losses = Losses()
        self.optimizers = Optimizers()
