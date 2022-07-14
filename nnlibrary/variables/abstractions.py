from abc import ABC
from abc import abstractmethod

import numpy as np

from nnlibrary.layer_structures import AbstractLayerStructure


class AbstractVariables(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def init_variables(self, layer_structure: AbstractLayerStructure):
        pass

    @abstractmethod
    def set_all(self, value: np.ndarray):
        pass

    @abstractmethod
    def set_single(self, layer_number: int, value: np.ndarray):
        pass

    @abstractmethod
    def get_all(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_single(self, layer_number: int) -> np.ndarray:
        pass
