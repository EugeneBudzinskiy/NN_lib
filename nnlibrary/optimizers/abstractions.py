from abc import ABC
from abc import abstractmethod

import numpy as np
from nnlibrary.variables import AbstractVariables


class AbstractOptimizer(ABC):
    @abstractmethod
    def __call__(self, trainable_variables: AbstractVariables, gradient_vector: np.ndarray):
        pass
