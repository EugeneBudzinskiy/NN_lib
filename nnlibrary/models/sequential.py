import numpy as np

from nnlibrary import initializers
from nnlibrary import layer_structures
from nnlibrary import layers
from nnlibrary import losses
from nnlibrary import optimizers
from nnlibrary import variables

from nnlibrary.auto_diff import AutoDiff
from nnlibrary.models import AbstractModel


class Sequential(AbstractModel):
    def __init__(self):
        self.is_compiled = False

        self.layer_structure = layer_structures.LayerStructure()
        self.trainable_variables = variables.TrainableVariables()

        # noinspection PyTypeChecker
        self.optimizer: optimizers.AbstractOptimizer = None
        # noinspection PyTypeChecker
        self.loss: losses.AbstractLoss = None

    def get_variables(self) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.trainable_variables.get_all()

    def add(self, layer: layers.AbstractLayer):
        if self.is_compiled:
            raise Exception()  # TODO Custom Exception (not changeable after compile)

        self.layer_structure.add_layer(layer=layer)

    def compile(self,
                optimizer: optimizers.AbstractOptimizer = None,
                loss: losses.AbstractLoss = None,
                weight_initializer: initializers.AbstractInitializer = None,
                bias_initializer: initializers.AbstractInitializer = None):

        if self.is_compiled:
            raise Exception()  # TODO Custom Exception (already compiled)

        self.optimizer = optimizers.SGD() if optimizer is None else optimizer
        self.loss = losses.MeanSquaredError() if loss is None else loss

        self.trainable_variables.init_variables(
            layer_structure=self.layer_structure,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )
        self.is_compiled = True

    def _unpack_variables(self, layer_number: int) -> (np.ndarray, np.ndarray):
        current_layer = self.layer_structure.get_layer(layer_number=layer_number)
        current_vars = self.trainable_variables.get_single(layer_number=layer_number)

        current_node_count = current_layer.node_count
        previous_node_count = (len(current_vars) // current_node_count) - 1
        w_size = previous_node_count * current_node_count

        current_weight = current_vars[:w_size].reshape((previous_node_count, current_node_count))
        current_bias = current_vars[w_size:].reshape((1, -1))

        return current_weight, current_bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        a = x.copy()
        for i in range(1, self.layer_structure.layers_number):
            current_layer = self.layer_structure.get_layer(layer_number=i)
            current_weight, current_bias = self._unpack_variables(layer_number=i)

            if not isinstance(current_layer, layers.AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            a = current_layer.activation(x=np.dot(a, current_weight) + current_bias)
        return a

    def _model_wrapper(self, inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray):
        self.trainable_variables.set_all(value=weights)
        y_predicted = self.predict(x=inputs)
        return self.loss(y_predicted=y_predicted, y_target=targets)

    def backpropagation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        value = AutoDiff.backward_mode.gradient(
            func=lambda w: self._model_wrapper(inputs=x, targets=y, weights=w),
            x=self.trainable_variables.get_all()
        )
        self.trainable_variables.set_all(
            value=AutoDiff.backward_mode.value_to_numpy(x=self.trainable_variables.get_all())
        )
        return value

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            shuffle: bool = True):

        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        size = x.shape[0]
        indexes = np.arange(size)
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indexes)

            for i in range(0, size, batch_size):
                idx = indexes[i:i + batch_size]
                gradient_vector = self.backpropagation(x=x[idx], y=y[idx])
                adjustment = self.optimizer(gradient_vector=gradient_vector)
                self.trainable_variables.set_all(value=self.trainable_variables.get_all() + adjustment)
