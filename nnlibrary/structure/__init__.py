import numpy as np

from nnlibrary.layers.LayerTypes import Layer
from nnlibrary.layers.LayerTypes import ActivationLayer
from nnlibrary.errors import WrongStructure
from nnlibrary.errors import WrongStructureElement


class Structure:
    def __init__(self, structure: tuple):
        if len(structure) > 1:
            previous_node_count = 0
            variable_size = 0

            node_counts_list = list()
            variable_map_list = list()

            activations_function_list = list()
            activations_derivative_list = list()

            for layer in structure:
                if isinstance(layer, Layer):
                    current_node_count = layer.get_node_count()

                    if isinstance(layer, ActivationLayer):
                        weight_end = variable_size + previous_node_count * current_node_count
                        bias_end = weight_end + current_node_count

                        variable_map_list.append((variable_size, weight_end, bias_end))
                        variable_size = bias_end

                        activations_function_list.append(layer.activation.activation)
                        activations_derivative_list.append(layer.activation.derivative)

                    node_counts_list.append(current_node_count)
                    previous_node_count = current_node_count
                else:
                    raise WrongStructureElement(layer)

            self.variables = np.zeros(variable_size)
            self.variables_map = tuple(variable_map_list)
            self.activations_function = tuple(activations_function_list)
            self.activations_derivative = tuple(activations_derivative_list)
            self.node_counts = tuple(node_counts_list)
            self.layer_count = len(self.node_counts)

        else:
            raise WrongStructure
