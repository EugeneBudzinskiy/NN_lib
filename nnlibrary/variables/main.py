from numpy import ndarray
from numpy import random

from nnlibrary.variables.abstractions import AbstractTrainableVariables
from nnlibrary.layer_structures import AbstractLayerStructure


class TrainableVariables(AbstractTrainableVariables):
    def __init__(self):
        # TODO Select more suitable approach (large 1d array + representation)
        self.weights = list()
        self.biases = list()

    def init_variables(self, layer_structure: AbstractLayerStructure):
        # TODO Write better initialization method or make several and let to chose
        layer_number = layer_structure.get_layers_number()
        for i in range(layer_number - 1):
            prev_layer = layer_structure.get_layer(layer_number=i)
            next_layer = layer_structure.get_layer(layer_number=i + 1)

            self.weights.append(random.random((prev_layer.node_count, next_layer.node_count)))
            self.biases.append(random.random((1, next_layer.node_count)))

    def update_variables(self, value):
        for i in range(len(value)):
            w, b = value[i]
            self.weights[i] += w
            self.biases[i] += b

    def get_weight(self, layer_number: int) -> ndarray:
        return self.weights[layer_number]

    def get_bias(self, layer_number: int) -> ndarray:
        return self.biases[layer_number]
