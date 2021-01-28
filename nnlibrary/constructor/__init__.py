from abc import ABC
from abc import abstractmethod

from nnlibrary.nn import NeuralNetwork
from nnlibrary.layers.LayerTypes import Layer
from nnlibrary.layers.LayerTypes import InputLayer
from nnlibrary.layers.LayerTypes import ActivationLayer
from nnlibrary.losses import Loss
from nnlibrary.optimizers import Optimizer

from nnlibrary.errors import InputLayerNotDefined
from nnlibrary.errors import InputLayerAlreadyDefined
from nnlibrary.errors import IsNotALayer


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
        self.structure = list()

    def show_structure(self):
        for el in self.structure:
            name = el.__class__.__name__
            node_count = el.get_node_count()
            print(f'{name} : {node_count}')

    def add(self, layer: Layer):
        if isinstance(layer, Layer):
            if isinstance(layer, InputLayer):
                if len(self.structure):
                    raise InputLayerAlreadyDefined
                else:
                    self.structure.append(layer)

            elif isinstance(layer, ActivationLayer):
                if len(self.structure):
                    self.structure.append(layer)
                else:
                    raise InputLayerNotDefined

            else:
                raise IsNotALayer(layer)
        else:
            raise IsNotALayer(layer)

    def compile(self, loss: Loss, optimizer: Optimizer) -> NeuralNetwork:
        pass
