from nnlibrary.layers import AbstractLayer
from .abstractions import AbstractLayerStructure


class LayerStructure(AbstractLayerStructure):
    def __init__(self):
        self.structure = list()

    def add_layer(self, layer: AbstractLayer):
        self.structure.append(layer)

    def get_layer(self, layer_number: int) -> AbstractLayer:
        return self.structure[layer_number]

    @property
    def layers_number(self) -> int:
        return len(self.structure)
