from Errors import InputLayerNotDefined
from NeuralNetwork import NeuralNetwork


class Constructor:
    def __init__(self):
        self.__structure = dict()
        self.__hidden_layer_counter = 0

        self.__input_layer_name = 'input_layer'
        self.__hidden_layer_name = 'hidden_layer'
        self.__output_layer_name = 'output_layer'

        self.__input_init_flag = False
        self.__output_init_flag = False

    def show_structure(self):
        for key, value in self.__structure.items():
            raw = value[-1]
            func_name = raw if isinstance(raw, str) else raw.__name__
            print(f'{key:18}: {str(value[0]):6} {func_name}')

    def add_input(self, node_count: int):
        self.__structure[self.__input_layer_name] = (node_count, '...')
        self.__input_init_flag = True

    def add_layer(self, node_count: int, activation_func):
        if self.__input_init_flag:
            if self.__output_init_flag:
                self.__hidden_layer_counter += 1
                new_hidden_name = f'{self.__hidden_layer_name}_{self.__hidden_layer_counter}'
                self.__structure[new_hidden_name] = self.__structure.popitem()[-1]
                self.__structure[self.__output_layer_name] = (node_count, activation_func)

            else:
                self.__structure[self.__output_layer_name] = (node_count, activation_func)
                self.__output_init_flag = True

        else:
            raise InputLayerNotDefined

    def compile(self, loss_function, optimizer) -> NeuralNetwork:
        pass
