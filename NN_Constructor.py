import numpy as np
from abc import ABC
from abc import abstractmethod

from NeuralNetwork import NeuralNetwork
from Functions import Functions
from Errors import InputLayerNotDefined
from Errors import WrongStructure


class AbstractConstructor(ABC):
    @abstractmethod
    def show_structure(self):
        pass

    @abstractmethod
    def add_input(self, node_count):
        pass

    @abstractmethod
    def add_layer(self, node_count, activation_func):
        pass

    @abstractmethod
    def compile(self, loss_function, optimizer) -> NeuralNetwork:
        pass


class Constructor(AbstractConstructor):
    def __init__(self):
        self.__f = Functions()
        self.__structure = dict()
        self.__hidden_layer_counter = 0

        self.__input_layer_name = 'input_layer'
        self.__hidden_layer_name = 'hidden_layer'
        self.__output_layer_name = 'output_layer'
        self.__empty_activation_name = '...'

        self.__input_init_flag = False
        self.__output_init_flag = False

    def show_structure(self):
        for key, value in self.__structure.items():
            raw = value[-1]
            func_name = raw if isinstance(raw, str) else raw.__name__
            print(f'{key:18}: {str(value[0]):6} {func_name}')

    def add_input(self, node_count: int):
        self.__structure[self.__input_layer_name] = (node_count, self.__empty_activation_name)
        self.__input_init_flag = True

    def add_layer(self, node_count: int, activation_func: id):
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

    def __parse_nn_structure(self):
        node_count_list = list()
        act_function_list = list()
        act_function_der_list = list()

        for value in self.__structure.values():
            node_count, act_function = value
            node_count_list.append(node_count)

            if act_function != self.__empty_activation_name:
                act_function_list.append(act_function)
                act_function_der = self.__f.get_act_func_der(act_function)
                act_function_der_list.append(act_function_der)

        return tuple(node_count_list), tuple(act_function_list), tuple(act_function_der_list)

    @staticmethod
    def __init_var_map(node_count: tuple, layer_count: int):
        var_map = list()
        position = 0

        for i in range(layer_count - 1):
            prev_nc = node_count[i]
            next_nc = node_count[i + 1]

            weights_size = prev_nc * next_nc
            biases_size = next_nc

            start = position
            end_weights = start + weights_size
            end_biases = end_weights + biases_size

            var_map.append((start, end_weights, end_biases))
            position = end_biases

        return tuple(var_map)

    @staticmethod
    def __init_variables(var_map: tuple):
        variables = np.zeros(var_map[-1][-1])
        for el in var_map:
            start, end, _ = el
            w_size = end - start
            current_weight = np.random.randn(w_size)  # TODO Maybe, change init of weights >:)
            variables[start:end] = current_weight
        return variables

    def __reset_all(self):
        self.__structure = dict()
        self.__hidden_layer_counter = 0
        self.__input_init_flag = False
        self.__output_init_flag = False

    def compile(self, loss_func: id, optimizer: id) -> NeuralNetwork:
        if not self.__input_init_flag or not self.__output_init_flag:
            raise WrongStructure

        else:
            node_count, activation_func, activation_func_der = self.__parse_nn_structure()
            layer_count = len(node_count)
            var_map = self.__init_var_map(node_count=node_count, layer_count=layer_count)

            variables = self.__init_variables(var_map)
            loss_func_der = self.__f.get_loss_func_der(loss_func)

            self.__reset_all()
            return NeuralNetwork(variables=variables,
                                 var_map=var_map,
                                 node_count=node_count,
                                 activation_func=activation_func,
                                 activation_func_der=activation_func_der,
                                 loss_func=loss_func,
                                 loss_func_der=loss_func_der,
                                 optimizer=optimizer,
                                 layer_count=layer_count)
