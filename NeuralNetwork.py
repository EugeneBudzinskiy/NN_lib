import numpy as np

from Functions import Func


class NeuralNetwork:
    def __init__(self):
        self.__input_layer_name = 'input_layer'
        self.__hidden_layer_name = 'hidden_layer'
        self.__output_layer_name = 'output_layer'

        self.__compile_flag = False
        self.__hidden_layer_count = 0

        self.__structure = dict()
        self.functions = tuple()
        self.layers_sizes = tuple()

        self.__func = Func()
        self.variables = np.array([])

    def add_layer(self, node_count: int, activation_func: str = ''):
        if self.__input_layer_name in self.__structure:

            if self.__output_layer_name in self.__structure:
                self.__hidden_layer_count += 1
                hidden_layer_name = f'{self.__hidden_layer_name}_{self.__hidden_layer_count}'

                self.__structure[hidden_layer_name] = self.__structure.popitem()[-1]
                self.__structure[self.__output_layer_name] = (node_count, activation_func)  # Create new 'Output Layer'

            else:
                self.__structure[self.__output_layer_name] = (node_count, activation_func)

        else:
            self.__structure[self.__input_layer_name] = (node_count, 'no activation')

    def show_structure(self):
        for key, value in self.__structure.items():
            print(f'{key:18}: {str(value[0]):6} {value[-1]}')

    def __check_not_compiled_error(self):
        if not self.__compile_flag:
            raise NotImplementedError("NN doesnt compile yet")  # TODO MB Replace

    def __check_already_compiled_error(self):
        if self.__compile_flag:
            raise NotImplementedError("NN already compile yet")  # TODO MB Replace

    def compile(self):
        self.__check_already_compiled_error()

        if self.__input_layer_name in self.__structure and self.__output_layer_name in self.__structure:
            current_functions = list()
            current_layers_sizes = list()

            size_variables = 0
            last_layer_node_count = self.__structure[self.__input_layer_name][0]
            current_layers_sizes.append(last_layer_node_count)

            for i in range(self.__hidden_layer_count):
                current_hid_name = f'{self.__hidden_layer_name}_{i + 1}'
                current_layer_data = self.__structure[current_hid_name]
                current_layer_node_count = current_layer_data[0]

                current_layers_sizes.append(current_layer_node_count)
                current_functions.append(self.__func.get(current_layer_data[-1]))

                size_variables += (last_layer_node_count + 1) * current_layer_node_count
                last_layer_node_count = current_layer_node_count

            output_layer_data = self.__structure[self.__output_layer_name]
            output_layer_node_count = output_layer_data[0]

            current_layers_sizes.append(output_layer_node_count)
            current_functions.append(self.__func.get(output_layer_data[-1]))

            self.layers_sizes = tuple(current_layers_sizes)
            self.functions = tuple(current_functions)

            size_variables += (last_layer_node_count + 1) * output_layer_node_count
            self.variables = np.zeros(size_variables)

            self.__compile_flag = True
        else:
            raise NotImplementedError("NN not compilable")  # TODO MB later rewrite by own Exception

    def predict(self, data_x):
        self.__check_not_compiled_error()

        current_data = data_x
        position = 0

        for i in range(self.__hidden_layer_count + 1):
            last_layer_size = self.layers_sizes[i]
            next_layer_size = self.layers_sizes[i + 1]

            end_weights = position + last_layer_size * next_layer_size
            end_biases = end_weights + next_layer_size

            weights = self.variables[position:end_weights].reshape(last_layer_size, next_layer_size)
            biases = self.variables[end_weights:end_biases]

            activation_func = self.functions[i][0]
            current_data = activation_func(np.dot(current_data, weights) + biases)

            position = end_biases

        return current_data
