import numpy as np
from abc import ABC
from abc import abstractmethod

from nnlibrary.nn import NeuralNetwork
from nnlibrary.layers.LayerTypes import Layer
from nnlibrary.layers.LayerTypes import InputLayer
from nnlibrary.layers.LayerTypes import ActivationLayer
from nnlibrary.optimizers import Optimizer
from nnlibrary.structure import Structure
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
        tuple_structure = tuple(self.structure)
        self.structure = list()

        compiled_structure = Structure(tuple_structure)
        self.weight_initialization(compiled_structure)

        return NeuralNetwork(structure=compiled_structure, loss=loss, optimizer=optimizer)

    @staticmethod
    def weight_initialization(structure: Structure):
        for i in range(structure.layer_count - 1):
            weight_start, weight_end, _ = structure.variables_map[i]

            previous_node = structure.node_counts[i]
            next_node = structure.node_counts[i + 1]

            coefficient = np.sqrt(2 / (previous_node + next_node))

            structure.variables[weight_start:weight_end] = coefficient * np.random.rand(weight_end - weight_start)