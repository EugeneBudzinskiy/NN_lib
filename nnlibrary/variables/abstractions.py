from abc import ABC
from abc import abstractmethod

from numpy import ndarray

from nnlibrary.layer_structures import AbstractLayerStructure


class AbstractTrainableVariables(ABC):  # TODO Add init (+ add restriction to variable inner structure)
    @abstractmethod
    def init_variables(self, layer_structure: AbstractLayerStructure):
        pass

    @abstractmethod
    def update_variables(self, value):  # TODO Replace with setter
        pass

    @abstractmethod
    def get_weight(self, layer_number: int) -> ndarray:
        pass

    @abstractmethod
    def get_bias(self, layer_number: int) -> ndarray:
        pass
