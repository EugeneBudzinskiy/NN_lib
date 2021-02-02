import numpy as np


class VariableStorage:
    def __init__(self, structure: tuple):
        self.variables = None
        self.map = None
        self.shape_map = None

        self._initialization(structure)

    def _initialization(self, structure: tuple):
        map_buffer = list()
        shape_map_buffer = list()

        previous_node_count = structure[0].node_count
        position = 0

        for i in range(1, len(structure)):
            current_node_count = structure[i].node_count

            weight_end = position + previous_node_count * current_node_count
            bias_end = weight_end + current_node_count * int(structure[i].bias_flag)

            map_buffer.append((position, weight_end, bias_end))
            shape_map_buffer.append((previous_node_count, current_node_count))

            previous_node_count = current_node_count
            position = bias_end

        self.variables = np.zeros(position)
        self.map = tuple(map_buffer)
        self.shape_map = tuple(shape_map_buffer)

        self._weight_initialization(structure)

    def _weight_initialization(self, structure: tuple):
        for i in range(len(structure) - 1):
            position, weight_end, _ = self.map[i]
            previous_node_count, current_node_count = self.shape_map[i]

            coefficient = np.sqrt(1 / (previous_node_count + current_node_count))
            self.variables[position:weight_end] = coefficient * np.random.randn(weight_end - position)
