import numpy as np

from nnlibrary.variables.abstractions import AbstractVariables
from nnlibrary.layer_structures import AbstractLayerStructure


class TrainableVariables(AbstractVariables):
    def __init__(self):
        # TODO Select more suitable approach (large 1d array + representation)
        self.weights = list()
        self.biases = list()

        self.__variables = np.array([])
        self.__map = np.array([])

    @staticmethod
    def _calculate_sizes(layer_structure: AbstractLayerStructure) -> (int, int):
        layers_number = layer_structure.get_layers_number()
        variable_size, map_size = 0, layers_number - 1
        for i in range(layers_number - 1):
            variable_size += layer_structure.get_layer(layer_number=i).node_count

        return variable_size, map_size

    def init_variables(self, layer_structure: AbstractLayerStructure):
        # TODO Write better initialization method or make several and let to chose
        variable_size, map_size = self._calculate_sizes(layer_structure=layer_structure)
        self.__variables, self.__map = np.zeros()
        layer_number = layer_structure.get_layers_number()
        for i in range(layer_number - 1):
            prev_layer = layer_structure.get_layer(layer_number=i)
            next_layer = layer_structure.get_layer(layer_number=i + 1)

            self.weights.append(np.random.random((prev_layer.node_count, next_layer.node_count)))
            self.biases.append(np.random.random((1, next_layer.node_count)))

    def set_all(self, value: np.ndarray):
        pass

    def set_single(self, layer_number: int, value: np.ndarray):
        pass

    def get_all(self) -> np.ndarray:
        pass

    def get_single(self, layer_number: int) -> np.ndarray:
        pass

