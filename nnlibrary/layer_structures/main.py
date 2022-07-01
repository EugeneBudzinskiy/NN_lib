from nnlibrary.layer_structures import AbstractLayerStructure
from nnlibrary.layers import AbstractLayer


class LayerStructure(AbstractLayerStructure):
    def __init__(self):
        self.structure = list()

    def add_layer(self, layer: AbstractLayer):
        self.structure.append(layer)

    def get_layer(self, layer_number: int) -> AbstractLayer:
        return self.structure[layer_number]

    def get_layers_number(self) -> int:
        return len(self.structure)
