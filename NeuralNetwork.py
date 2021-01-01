import numpy as np

from Functions import Func


class NeuralNetwork:
    def __init__(self):
        self.__func = Func()  # Singleton with all function for NN

        self.__structure = {}  # Dictionary for structure of NN
        self.__hidden_layer_count = 0  # Counter of 'Hidden Layers' of NN, for easy naming of layers

        self.__input_layer_name = 'input_layer'
        self.__hidden_layer_name = 'hidden_layer'
        self.__output_layer_name = 'output_layer'

        self.__compile_flag = False  # Simple flag for tracking compile status

        self.variables = None  # NumPy array with all variables of NN (weights and biases)

        self.functions = list()  # List with pair of '(func, func_derivative)'

    def add_layer(self, node_count: int, activation_func: str = ''):
        if self.__input_layer_name in self.__structure:  # Check if 'Input Layer' added to 'structure'
            if self.__output_layer_name in self.__structure:  # Check if 'Output Layer' added to 'structure'
                self.__hidden_layer_count += 1  # Count of existing 'Hidden Layers' in 'structure'
                hidden_layer_name = \
                    f'{self.__hidden_layer_name}_{self.__hidden_layer_count}'  # Create name of new 'Hidden Layer'

                self.__structure[hidden_layer_name] = \
                    self.__structure.popitem()[-1]  # Rewrite 'Output Layer' data to new 'Hidden Layer'
                self.__structure[self.__output_layer_name] = (node_count, activation_func)  # Create new 'Output Layer'

            else:  # if in 'structure' exist only 'Input Layer' then add 'Output Layer'
                self.__structure[self.__output_layer_name] = (node_count, activation_func)

        else:  # if 'structure' is empty, add 'Input Layer'
            self.__structure[self.__input_layer_name] = (node_count, 'no activation')

    def show_structure(self):  # Just printing of 'structure' dictionary
        for key, value in self.__structure.items():
            print(f'{key:18}: {str(value[0]):6} {value[-1]}')

    def __check_not_compiled_error(self):
        if not self.__compile_flag:  # If not compiled, raise Error
            raise NotImplementedError("NN doesnt compile yet")  # TODO MB Replace

    def __check_already_compiled_error(self):
        if self.__compile_flag:  # If already compiled, raise Error
            raise NotImplementedError("NN already compile yet")  # TODO MB Replace

    def compile(self):  # Compile process of NN
        self.__check_already_compiled_error()  # Check if NN already compile

        if self.__input_layer_name in self.__structure \
                and self.__output_layer_name in self.__structure:  # Check how right the structure of NN
            size_weights = 0  # Number of all 'weights' of NN
            size_biases = 0  # Number of all 'biases' of NN

            last_layer_node_count = self.__structure[self.__input_layer_name][0]  # Get number of nodes in 'Input Layer'

            for i in range(self.__hidden_layer_count):  # Run throw all 'Hidden layer' of NN
                current_hid_name = f'{self.__hidden_layer_name}_{i + 1}'  # Create name of current 'Hidden Layer'
                current_layer_data = self.__structure[current_hid_name]  # Get data from current 'Hidden Layer'
                current_layer_node_count = current_layer_data[0]  # Get number of nodes in current 'Hidden Layer'
                self.functions.append(self.__func.get(current_layer_data[-1]))  # Get name of 'activation function'

                size_weights += last_layer_node_count * current_layer_node_count  # Increase size of 'weights'
                size_biases += current_layer_node_count  # Increase size of 'biases'

                last_layer_node_count = current_layer_node_count  # Update 'new' node count of last layer

            output_layer_data = self.__structure[self.__output_layer_name]  # Get data from 'Output Layer'
            output_layer_node_count = output_layer_data[0]  # Get number of nodes in 'Output Layer'
            self.functions.append(self.__func.get(output_layer_data[-1]))  # Get name of 'activation function'

            size_weights += last_layer_node_count * output_layer_node_count  # Increase size of 'weights'
            size_biases += output_layer_node_count  # Increase size of 'biases'

            self.variables = np.zeros(10)

            self.__compile_flag = True  # When process end, set compile flag to True
        else:
            raise NotImplementedError("NN not compilable")  # TODO MB later rewrite by own Exception

    def predict(self, data_x):
        self.__check_not_compiled_error()  # Check if NN not compiled yet
