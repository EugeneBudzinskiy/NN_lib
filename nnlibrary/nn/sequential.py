from nnlibrary.layers.layer_types import InputLayer
from nnlibrary.layers.layer_types import Layer
from nnlibrary.errors import InputLayerAlreadyDefined
from nnlibrary.errors import IsNotALayer
from nnlibrary.errors import NothingToPop
from nnlibrary.errors import WrongLayerIndex
from nnlibrary.errors import ProvideLayerIndex
from nnlibrary.errors import WrongStructure


class Sequential:
    def __init__(self):
        self.input_layer = None
        self.layers = list()
        self.weight = list()
        self.bias = list()

    def add(self, layer):
        if isinstance(layer, Layer):
            if isinstance(layer, InputLayer):
                if self.input_layer is None:
                    self.input_layer = layer
                else:
                    raise InputLayerAlreadyDefined
            else:
                self.layers.append(layer)
        else:
            raise IsNotALayer(layer)

    def pop(self):
        if len(self.layers):
            self.layers.pop()
        else:
            raise NothingToPop

    def get_layer(self, index: int = None):
        if index is not None:
            if len(self.layers) <= index:
                raise WrongLayerIndex
            else:
                return self.layers[index]
        else:
            raise ProvideLayerIndex

    def show_structure(self):
        return self.layers

    def compile(self):
        if len(self.layers) > 0:
            pass
        else:
            raise WrongStructure

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_batch(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def summary(self):
        pass
