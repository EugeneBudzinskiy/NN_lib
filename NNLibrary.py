from SingletonMeta import SingletonMeta
from NNConstructor import Constructor
from functions import Functions


class NNLib(metaclass=SingletonMeta):
    def __init__(self):
        self.constructor = Constructor()
        self.functions = Functions()
