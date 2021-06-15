from nnlibrary.layers.layer_types import ActivationLayer
from nnlibrary.layers.layer_types import Layer
from nnlibrary.errors import InputLayerAlreadyDefined
from nnlibrary.errors import IsNotALayer
from nnlibrary.errors import NothingToPop
from nnlibrary.errors import WrongLayerIndex
from nnlibrary.errors import ProvideLayerIndex
from nnlibrary.errors import WrongStructure


class Sequential:
    def __init__(self):
        self._is_compiled = False

        self._input_layer = None
        self._layers = list()

        self._weight = list()
        self._bias = list()

    @property
    def layers(self):
        return self._layers

    @property
    def weights(self):
        return self._weight, self._bias

    def add(self, layer):
        if isinstance(layer, Layer):
            if isinstance(layer, ActivationLayer):
                self._layers.append(layer)
            else:
                if self._input_layer is None:
                    self._input_layer = layer
                else:
                    raise InputLayerAlreadyDefined
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
