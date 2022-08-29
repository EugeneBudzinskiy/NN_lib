import numpy as np

from nnlibrary.auto_diff import AbstractMode
from nnlibrary.auto_diff import AbstractSpecialVariable
from nnlibrary.auto_diff.reverse_mode import special_vars


class ReverseMode(AbstractMode):
    @staticmethod
    def to_variable(x: np.ndarray) -> np.ndarray:
        vec_f = np.vectorize(lambda val: special_vars.Variable(value=val))
        return vec_f(val=x)

    @staticmethod
    def to_variable_direction(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        vec_f = np.vectorize(lambda val, grad: special_vars.Variable(value=val, partial=grad))
        return vec_f(val=x, grad=vector)

    def apply_partials_backwards(self, inputs: tuple, partials: tuple, multiplier: float):
        for inp, par in zip(inputs, partials):
            self.set_partial(var_x=inp, value=multiplier * par)

    def backward_pass(self, final_node: AbstractSpecialVariable):
        current_node = final_node
        current_node.partial = 1.

        lifo_queue = list()
        flag = True
        while flag:
            if isinstance(current_node, special_vars.Variable):
                if len(lifo_queue):
                    current_node = lifo_queue.pop()
                else:
                    flag = False

            elif isinstance(current_node, special_vars.Operator):
                self.apply_partials_backwards(inputs=current_node.inputs,
                                              partials=current_node.inputs_partials,
                                              multiplier=current_node.partial)
                current_node = current_node.inputs[0]
                lifo_queue.extend(current_node.inputs[1:])

            else:
                raise Exception()  # TODO Custom Exception
