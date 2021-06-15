import numpy as np

from nnlibrary.layers.layer_types import ActivationLayer
from nnlibrary.layers.layer_types import Layer
from nnlibrary.errors import InputLayerAlreadyDefined
from nnlibrary.errors import InputLayerNotDefined
from nnlibrary.errors import IsNotALayer
from nnlibrary.errors import NothingToPop
from nnlibrary.errors import WrongLayerIndex
from nnlibrary.errors import ProvideLayerIndex
from nnlibrary.errors import WrongStructure
from nnlibrary.errors import AlreadyCompiled
from nnlibrary.errors import NotCompiled
from nnlibrary.errors import TryModifyCompiledNN


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
        if self._is_compiled:
            raise TryModifyCompiledNN
        else:
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
        if self._is_compiled:
            raise TryModifyCompiledNN
        else:
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

    @staticmethod
    def _weight_initialization(prev_nodes: int, curr_nodes: int):
        coefficient = np.sqrt(1 / (prev_nodes + curr_nodes))
        return (coefficient * np.random.randn(prev_nodes * curr_nodes)).reshape((prev_nodes, curr_nodes))

    def compile(self):
        if len(self._layers) <= 0:
            raise WrongStructure
        elif self._input_layer is None:
            raise InputLayerNotDefined
        else:
            if not self._is_compiled:
                prev_nodes = self._input_layer.node_count
                for el in self._layers:
                    curr_nodes = el.node_count

                    curr_weight = self._weight_initialization(prev_nodes=prev_nodes, curr_nodes=curr_nodes)
                    self._weight.append(curr_weight)
                    self._bias.append(np.zeros((1, prev_nodes)))

                    prev_nodes = curr_nodes

                self._is_compiled = True
            else:
                raise AlreadyCompiled

    def fit(self):
        pass

    def predict(self, x):
        if self._is_compiled:
            pass
        else:
            raise NotCompiled

    def predict_on_batch(self):
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
