from abc import ABC
from abc import abstractmethod

from nnlibrary.layers import AbstractLayer


class AbstractLayerStructure(ABC):
    @abstractmethod
    def add_layer(self, layer: AbstractLayer):
        pass

    @abstractmethod
    def get_layer(self, layer_number: int) -> AbstractLayer:
        pass

    @abstractmethod
    def get_layers_number(self) -> int:
        pass
