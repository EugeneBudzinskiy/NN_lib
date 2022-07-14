import numpy as np

from nnlibrary.variables.abstractions import AbstractVariables
from nnlibrary.layer_structures import AbstractLayerStructure


class TrainableVariables(AbstractVariables):
    def __init__(self):
        # TODO Select more suitable approach (large 1d array + representation)

        self.__variables = np.array([])
        self.__map = np.array([])

    def __len__(self):
        return len(self.__variables)

    def _set_inner_sizes(self, layer_structure: AbstractLayerStructure) -> (int, int):
        map_size = 3 * (layer_structure.layers_number - 1)
        variable_size = 0
        for i in range(1, layer_structure.layers_number):
            variable_size += (layer_structure.get_layer(layer_number=i - 1).node_count + 1) * \
                             layer_structure.get_layer(layer_number=i).node_count

        self.__variables, self.__map = np.zeros(variable_size), np.zeros(map_size, dtype=int)

    def __fill_map(self, layer_structure: AbstractLayerStructure):
        buff = 0
        for i in range(layer_structure.layers_number - 1):
            prev_node_count = layer_structure.get_layer(layer_number=i).node_count
            next_node_count = layer_structure.get_layer(layer_number=i + 1).node_count
            self.__map[3 * i + 0] = buff + 0
            self.__map[3 * i + 1] = buff + prev_node_count * next_node_count
            self.__map[3 * i + 2] = buff + (prev_node_count + 1) * next_node_count

            buff += (prev_node_count + 1) * next_node_count

    def __unpack_map_single(self, layer_number: int) -> (int, int, int):
        return self.__map[3 * layer_number + 0], \
               self.__map[3 * layer_number + 1], \
               self.__map[3 * layer_number + 2]

    def init_variables(self, layer_structure: AbstractLayerStructure):
        # TODO Write better initialization method or make several and let to chose
        self._set_inner_sizes(layer_structure=layer_structure)
        self.__fill_map(layer_structure=layer_structure)

        for i in range(layer_structure.layers_number - 1):
            w_s, b_s, b_e = self.__unpack_map_single(layer_number=i)
            self.__variables[w_s:b_s] = np.random.random(size=b_s - w_s)

    def set_all(self, value: np.ndarray):
        if value.shape == self.__variables.shape:
            self.__variables = value.copy()
        else:
            raise Exception()  # TODO Custom Exception

    def set_single(self, layer_number: int, value: np.ndarray):
        raise Exception('Implementation doesnt exist yet!')  # TODO Custom Exception

    def get_all(self) -> np.ndarray:
        return self.__variables.copy()

    def get_single(self, layer_number: int) -> np.ndarray:
        w_s, _, b_e = self.__unpack_map_single(layer_number=layer_number - 1)
        return self.__variables[w_s:b_e].copy()

