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

    def _set_inner_sizes(self, layer_structure: AbstractLayerStructure) -> (int, int):
        map_size = 3 * (layer_structure.layers_number - 1)
        variable_size = sum([layer_structure.get_layer(layer_number=i).node_count
                             for i in range(1, layer_structure.layers_number)])

        self.__variables, self.__map = np.zeros(variable_size), np.zeros(map_size)

    def _fill_map(self, layer_structure: AbstractLayerStructure):
        buff = 0
        for i in range(layer_structure.layers_number - 1):
            prev_node_count = layer_structure.get_layer(layer_number=i).node_count
            next_node_count = layer_structure.get_layer(layer_number=i + 1).node_count
            self.__map[3 * i + 0] = buff + 0
            self.__map[3 * i + 1] = buff + prev_node_count * next_node_count
            self.__map[3 * i + 2] = buff + prev_node_count * next_node_count + next_node_count

            buff += (prev_node_count + 1) * next_node_count

    def init_variables(self, layer_structure: AbstractLayerStructure):
        # TODO Write better initialization method or make several and let to chose
        self._set_inner_sizes(layer_structure=layer_structure)
        self._fill_map(layer_structure=layer_structure)

        print(self.__map)
        exit(-22)


        # layer_number = layer_structure.get_layers_number()
        # for i in range(layer_number - 1):
        #     prev_layer = layer_structure.get_layer(layer_number=i)
        #     next_layer = layer_structure.get_layer(layer_number=i + 1)
        #
        #     self.weights.append(np.random.random((prev_layer.node_count, next_layer.node_count)))
        #     self.biases.append(np.random.random((1, next_layer.node_count)))

    def set_all(self, value: np.ndarray):
        pass

    def set_single(self, layer_number: int, value: np.ndarray):
        pass

    def get_all(self) -> np.ndarray:
        pass

    def get_single(self, layer_number: int) -> np.ndarray:
        pass

