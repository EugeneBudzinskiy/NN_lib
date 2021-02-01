from abc import ABC
from abc import abstractmethod

from nnlibrary.nn import NeuralNetwork
from nnlibrary.layers.LayerTypes import Layer
from nnlibrary.layers.LayerTypes import InputLayer
from nnlibrary.layers.LayerTypes import ActivationLayer
from nnlibrary.optimizers import Optimizer
from nnlibrary.losses import Loss

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
    def compile(self, loss: Loss, optimizer: Optimizer) -> NeuralNetwork:
        pass


class Constructor(AbstractConstructor):
    def __init__(self):
        self.structure = list()

    def show_structure(self):
        for el in self.structure:
            name = el.__class__.__name__
            node_count = el.node_count
            print(f'{name} : {node_count}')

    def add(self, layer: Layer):
        if isinstance(layer, Layer):
            if isinstance(layer, InputLayer):
                if len(self.structure) > 0:
                    raise InputLayerAlreadyDefined
                else:
                    self.structure.append(layer)

            elif isinstance(layer, ActivationLayer):
                if len(self.structure) > 0:
                    layer.layer_initialization(self.structure[-1])
                    self.structure.append(layer)
                else:
                    raise InputLayerNotDefined

            else:
                raise IsNotALayer(layer)
        else:
            raise IsNotALayer(layer)

    def compile(self, loss: Loss, optimizer: Optimizer) -> NeuralNetwork:
        buffer = tuple(self.structure)
        self.structure = list()
        return NeuralNetwork(structure=buffer, loss=loss, optimizer=optimizer)
