from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class AbstractTrainableVariables(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init_variables(self):
        pass

    @abstractmethod
    def get_weight(self, layer_number: int) -> ndarray:
        pass

    @abstractmethod
    def get_bias(self, layer_number: int) -> ndarray:
        pass
