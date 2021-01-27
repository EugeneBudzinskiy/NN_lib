from singleton import SingletonMeta
from constructor import Constructor
from functions import Functions
from layers import Layers


class NNLib(metaclass=SingletonMeta):
    def __init__(self):
        self.constructor = Constructor()
        self.functions = Functions()
        self.layers = Layers()
