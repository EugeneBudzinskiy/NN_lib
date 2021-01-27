import numpy as np
from abc import ABC
from abc import abstractmethod

from NeuralNetwork import NeuralNetwork
from Layers import Layer
from Layers import InputLayer
from Layers import ActivationLayer

from Errors import InputLayerNotDefined
from Errors import InputLayerAlreadyDefined
from Errors import IsNotALayer


class AbstractConstructor(ABC):
    @abstractmethod
    def show_structure(self):
        pass

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def compile(self, loss, optimizer) -> NeuralNetwork:
        pass


class Constructor(AbstractConstructor):
    def __init__(self):
        self.flag_input_init = False
        self.flag_output_init = False

        self.structure = list()

    def show_structure(self):
        for el in self.structure:
            print(el)

    def add(self, layer: Layer):
        if isinstance(layer, Layer):

            if isinstance(layer, InputLayer):

                if self.flag_input_init:
                    raise InputLayerAlreadyDefined

                else:
                    self.structure.append(layer)
                    self.flag_input_init = True

            elif isinstance(layer, ActivationLayer):

                if self.flag_input_init:
                    self.flag_output_init = True
                    self.structure.append(layer)

                else:
                    raise InputLayerNotDefined

            else:
                raise IsNotALayer(layer)
        else:
            raise IsNotALayer(layer)

    def compile(self, loss, optimizer) -> NeuralNetwork:
        pass
