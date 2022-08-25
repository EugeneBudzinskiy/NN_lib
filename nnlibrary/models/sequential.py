import numpy as np

from nnlibrary import activations
from nnlibrary import differentiators
from nnlibrary import initializers
from nnlibrary import layer_structures
from nnlibrary import layers
from nnlibrary import losses
from nnlibrary import optimizers
from nnlibrary import variables
from nnlibrary.models import AbstractModel
from nnlibrary.reductions import ReductionNone


class Sequential(AbstractModel):
    def __init__(self):
        self.layer_structure = layer_structures.LayerStructure()

        # noinspection PyTypeChecker
        self.core: SequentialCompiledCore = None

    @property
    def is_compiled(self) -> bool:
        return isinstance(self.core, SequentialCompiledCore)

    def get_variables(self) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.trainable_variables.get_all()

    @property
    def loss(self) -> losses.AbstractLoss:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.loss

    @property
    def optimizer(self) -> optimizers.AbstractOptimizer:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        return self.core.optimizer

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

        optimizer = optimizers.SGD() if optimizer is None else optimizer
        loss = losses.MeanSquaredError() if loss is None else loss

        self.core = SequentialCompiledCore(
            layer_structure=self.layer_structure,
            optimizer=optimizer,
            loss=loss,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )

    def feedforward(self, x: np.ndarray) -> (np.ndarray, [np.ndarray], [np.ndarray]):
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        a = x.copy()
        z_list, a_list = list(), list()
        a_list.append(a)

        for i in range(1, self.layer_structure.layers_number):
            current_layer = self.layer_structure.get_layer(layer_number=i)
            current_weight, current_bias = self.core.unpack_variables(layer_number=i)

            z = np.dot(a, current_weight) + current_bias
            z_list.append(z)

            if not isinstance(current_layer, layers.AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            a = current_layer.activation(x=z)
            a_list.append(a)

        return a_list.pop(), z_list, a_list

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        output, _, _ = self.feedforward(x=x)
        return output

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        if not self.is_compiled:
            raise Exception()  # TODO Custom Exception (not compiled)

        output, z_list, a_list = self.feedforward(x=x)
        layers_number = self.layer_structure.layers_number

        loss_gradient = self.core.loss_gradient(y_target=y, y_predicted=output)
        derivative = self.core.activation_derivatives[-1](x=z_list[-1])
        delta = self.core.delta_multiplications[-1](x1=loss_gradient, x2=derivative)

        d_weight = np.dot(a_list[-1].T, delta)
        d_bias = np.sum(delta, axis=0).reshape(1, -1)

        gradient_list = list()
        gradient_list.append(d_bias)
        gradient_list.append(d_weight)

        for i in range(1, layers_number - 1):
            j = layers_number - i - 1
            previous_weight, _ = self.core.unpack_variables(layer_number=j + 1)

            next_delta = np.dot(delta, previous_weight.T)
            derivative = self.core.activation_derivatives[j](x=z_list[j - 1])
            delta = self.core.delta_multiplications[j](x1=next_delta, x2=derivative)

            d_weight = np.dot(a_list[j - 1].T, delta)
            d_bias = np.sum(delta, axis=0).reshape(1, -1)

            gradient_list.append(d_bias)
            gradient_list.append(d_weight)

        gradient_list.reverse()
        gradient_vector = np.concatenate(gradient_list, axis=None) / y.shape[0]
        return gradient_vector

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
                adjustment = self.core.optimizer(gradient_vector=gradient_vector)

                self.core.trainable_variables.set_all(
                    value=self.core.trainable_variables.get_all() + adjustment
                )


class SequentialCompiledCore:
    def __init__(self,
                 layer_structure: layer_structures.AbstractLayerStructure,
                 optimizer: optimizers.AbstractOptimizer,
                 loss: losses.AbstractLoss,
                 weight_initializer: initializers.AbstractInitializer,
                 bias_initializer: initializers.AbstractInitializer):

        self.derivative = differentiators.Derivative()
        self.gradient = differentiators.Gradient()

        self.trainable_variables = variables.TrainableVariables()
        self.layer_structure = layer_structure

        self.optimizer = optimizer
        self.loss = loss

        self.trainable_variables.init_variables(
            layer_structure=self.layer_structure,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )

        self.loss_gradient = self.get_loss_gradient()
        self.activation_derivatives = self.get_activation_derivatives()
        self.delta_multiplications = self.get_delta_multiplications()

    def get_loss_gradient(self) -> callable:
        def loss_wrapper(y_target: np.ndarray):
            return lambda x: self.loss(y_predicted=x, y_target=y_target, reduction=ReductionNone())

        if isinstance(self.loss, losses.CategoricalCrossentropy):

            if self.loss.from_logits:
                def result_gradient_func(y_predicted: np.ndarray, y_target: np.ndarray) -> np.ndarray:
                    softmax = activations.Softmax()
                    s = softmax(x=y_predicted)
                    tmp = np.repeat(s, s.shape[-1], axis=0)
                    matrix = - tmp * tmp.T + np.diag(s.ravel())
                    return np.dot(- y_target / s, matrix)

            else:
                def result_gradient_func(y_predicted: np.ndarray, y_target: np.ndarray) -> np.ndarray:
                    s = np.sum(y_predicted)
                    tmp = np.repeat(y_predicted, y_predicted.shape[-1], axis=0).T
                    diff = s - y_predicted
                    matrix = - tmp + np.diag(y_predicted.ravel()) + np.diag(diff.ravel())
                    return np.dot(- y_target / y_predicted, matrix / s)

            return lambda y_predicted, y_target: \
                result_gradient_func(y_predicted=y_predicted, y_target=y_target)

        else:
            return lambda y_target, y_predicted: \
                self.gradient(func=loss_wrapper(y_target=y_target), x=y_predicted)

    def get_activation_derivatives(self) -> [callable]:
        derivatives = list([None])
        for i in range(1, self.layer_structure.layers_number):
            if None:
                raise Exception('empty')

            else:
                current_layer = self.layer_structure.get_layer(layer_number=i)

                if not isinstance(current_layer, layers.AbstractActivationLayer):
                    raise Exception()  # TODO Custom Exception

                activation = current_layer.activation
                derivatives.append(lambda x: self.derivative(func=activation, x=x))

        return derivatives

    def get_delta_multiplications(self) -> [callable]:
        def normal_mul(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            return np.multiply(x1, x2)

        def jacobian_mul(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            result = np.empty_like(x1)
            for k in range(result.shape[0]):
                result[k] = np.dot(x1[k], x2[k])
            return result

        funcs = list([None])
        for i in range(1, self.layer_structure.layers_number):
            current_layer = self.layer_structure.get_layer(layer_number=i)

            if not isinstance(current_layer, layers.AbstractActivationLayer):
                raise Exception()  # TODO Custom Exception

            activation = current_layer.activation

            if isinstance(activation, activations.Softmax):
                funcs.append(jacobian_mul)
            else:
                funcs.append(normal_mul)

        return funcs

    def unpack_variables(self, layer_number: int) -> (np.ndarray, np.ndarray):
        current_layer = self.layer_structure.get_layer(layer_number=layer_number)
        current_vars = self.trainable_variables.get_single(layer_number=layer_number)

        current_node_count = current_layer.node_count
        previous_node_count = len(current_vars) // (current_node_count + 1)
        w_size = previous_node_count * current_node_count

        current_weight = current_vars[:w_size].reshape((previous_node_count, current_node_count))
        current_bias = current_vars[w_size:].reshape((1, -1))

        return current_weight, current_bias
