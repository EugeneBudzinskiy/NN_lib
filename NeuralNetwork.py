import numpy as np

from Functions import Func


class NeuralNetwork:
    def __init__(self):
        self.__func = Func()  # Singleton with all function for NN

        self.__structure = {}  # Dictionary for structure of NN
        self.__hidden_layer_count = 0  # Counter of 'Hidden Layers' of NN, for easy naming of layers

        self.__layer_names = {
            'input_layer_name': 'input_layer',
            'hidden_layer_name': 'hidden_layer',
            'output_layer_name': 'output_layer',
        }  # Dictionary with names of layer, for personalization of layers naming

        self.__compile_flag = False  # Simple flag for tracking compile status

        self.weights = None  # NumPy array with all weights of NN
        self.biases = None  # NumPy array with all biases of NN

        self.functions = list()  # List with pair of '(func, func_derivative)'

    def add_layer(self, node_count: int, activation_func: str):
        # Get all name, for easy access
        input_l_n = self.__layer_names['input_layer_name']
        hidden_l_n = self.__layer_names['hidden_layer_name']
        output_l_n = self.__layer_names['output_layer_name']

        if input_l_n in self.__structure:  # Check if 'Input Layer' added to 'structure'
            if output_l_n in self.__structure:  # Check if 'Output Layer' added to 'structure'
                self.__hidden_layer_count += 1  # Count of existing 'Hidden Layers' in 'structure'
                hidden_layer_name = \
                    f'{hidden_l_n}_{self.__hidden_layer_count}'  # Create name of new 'Hidden Layer'

                self.__structure[hidden_layer_name] = \
                    self.__structure.popitem()[-1]  # Rewrite 'Output Layer' data to new 'Hidden Layer'
                self.__structure[output_l_n] = (node_count, activation_func)  # Create new 'Output Layer'

            else:  # if in 'structure' exist only 'Input Layer' then add 'Output Layer'
                self.__structure[output_l_n] = (node_count, activation_func)

        else:  # if 'structure' is empty, add 'Input Layer'
            self.__structure[input_l_n] = (node_count, activation_func)

    def show_structure(self):  # Just printing of 'structure' dictionary
        for key, value in self.__structure.items():
            print(f'{key:18}: {str(value[0]):6} {value[-1]}')

    def compile(self):  # Compile process of NN
        if self.__compile_flag:  # If already compile, raise Error
            raise NotImplementedError("NN already compile")  # TODO MB Replace

        # Get all name, for easy access
        input_l_n = self.__layer_names['input_layer_name']
        hidden_l_n = self.__layer_names['hidden_layer_name']
        output_l_n = self.__layer_names['output_layer_name']

        if input_l_n in self.__structure and output_l_n in self.__structure:  # Check how right the structure of NN
            size_weights = 0  # Number of all 'weights' of NN
            size_biases = 0  # Number of all 'biases' of NN

            first_layer_data = self.__structure[input_l_n]  # Get data from 'Input Layer'
            last_layer_node_count = first_layer_data[0]  # Get number of nodes in 'Input Layer'
            self.functions.append(self.__func.get(first_layer_data[-1]))  # Get name of 'activation function'

            for i in range(self.__hidden_layer_count):  # Run throw all 'Hidden layer' of NN
                current_hid_name = f'{hidden_l_n}_{i + 1}'  # Create name of current 'Hidden Layer'
                current_layer_data = self.__structure[current_hid_name]  # Get data from current 'Hidden Layer'
                current_layer_node_count = current_layer_data[0]  # Get number of nodes in current 'Hidden Layer'
                self.functions.append(self.__func.get(current_layer_data[-1]))  # Get name of 'activation function'

                size_weights += last_layer_node_count * current_layer_node_count  # Increase size of 'weights'
                size_biases += current_layer_node_count  # Increase size of 'biases'

                last_layer_node_count = current_layer_node_count  # Update 'new' node count of last layer

            output_layer_data = self.__structure[output_l_n]  # Get data from 'Output Layer'
            output_layer_node_count = output_layer_data[0]  # Get number of nodes in 'Output Layer'
            self.functions.append(self.__func.get(output_layer_data[-1]))  # Get name of 'activation function'

            size_weights += last_layer_node_count * output_layer_node_count  # Increase size of 'weights'
            size_biases += output_layer_node_count  # Increase size of 'biases'

            self.weights = np.zeros(size_weights)  # Init NumPy array of weights in NN
            self.biases = np.zeros(size_biases)  # Init NumPy array of biases in NN

            self.__compile_flag = True  # When process end, set compile flag to True
        else:
            raise NotImplementedError("NN not compilable")  # TODO MB later rewrite by own Exception
