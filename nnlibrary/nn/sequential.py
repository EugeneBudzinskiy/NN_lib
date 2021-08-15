import time
import numpy as np
import nnlibrary as nnl


class Sequential:
    def __init__(self):
        self._is_compiled = False
        self._input_layer = None
        self._layers = list()

        self._variables = None
        self._optimizer = None
        self._loss = None

    @property
    def layers(self):
        return self._layers

    @property
    def weights(self):
        return self._variables

    def add(self, layer):
        if self._is_compiled:
            raise nnl.errors.TryModifyCompiledNN
        else:
            if isinstance(layer, nnl.layers.Layer):
                if isinstance(layer, nnl.layers.ActivationLayer):
                    self._layers.append(layer)
                else:
                    if self._input_layer is None:
                        self._input_layer = layer
                    else:
                        raise nnl.errors.InputLayerAlreadyDefined
            else:
                raise nnl.errors.IsNotALayer(layer)

    def pop(self):
        if self._is_compiled:
            raise nnl.errors.TryModifyCompiledNN
        else:
            if len(self.layers):
                self.layers.pop()
            else:
                raise nnl.errors.NothingToPop

    def get_layer(self, index: int = None):
        if index is not None:
            if len(self.layers) <= index:
                raise nnl.errors.WrongLayerIndex
            else:
                return self.layers[index]
        else:
            raise nnl.errors.ProvideLayerIndex

    def show_structure(self):
        return self._layers

    @staticmethod
    def _weight_initialization(prev_nodes: int, curr_nodes: int):
        coefficient = np.sqrt(1 / (prev_nodes + curr_nodes))
        return coefficient * np.random.randn(prev_nodes * curr_nodes)

    def _get_variables_count(self, stop: int = None):
        result = 0
        if stop is None:
            prev_node_count = self._input_layer.node_count
            for el in self._layers:
                curr_node_count = el.node_count
                result += (prev_node_count + 1) * curr_node_count
                prev_node_count = curr_node_count
        else:
            if 0 <= stop < len(self._layers):
                prev_node_count = self._input_layer.node_count
                for i in range(stop):
                    curr_node_count = self._layers[i].node_count
                    result += (prev_node_count + 1) * curr_node_count
                    prev_node_count = curr_node_count
            else:
                raise ValueError('`stop` out of range')
        return result

    def _get_weight(self, layer_number: int):
        if 0 <= layer_number < len(self._layers):
            if layer_number == 0:
                prev_node_count = self._input_layer.node_count
            else:
                prev_node_count = self._layers[layer_number - 1].node_count

            curr_node_count = self._layers[layer_number].node_count

            weight_pointer_start = self._get_variables_count(stop=layer_number)
            weight_pointer_end = weight_pointer_start + prev_node_count * curr_node_count
            return self._variables[weight_pointer_start:weight_pointer_end].reshape((prev_node_count, curr_node_count))
        else:
            raise ValueError('`layer_number` is out of range')

    def _get_bias(self, layer_number: int):
        if 0 <= layer_number < len(self._layers):
            if layer_number == 0:
                prev_node_count = self._input_layer.node_count
            else:
                prev_node_count = self._layers[layer_number - 1].node_count

            curr_node_count = self._layers[layer_number].node_count

            weight_pointer_start = self._get_variables_count(stop=layer_number) + prev_node_count * curr_node_count
            weight_pointer_end = weight_pointer_start + curr_node_count
            return self._variables[weight_pointer_start:weight_pointer_end].reshape((1, curr_node_count))
        else:
            raise ValueError('`layer_number` is out of range')

    def compile(self, optimizer=None, loss=None, metrics=None):
        if len(self._layers) <= 0:
            raise nnl.errors.WrongStructure
        elif self._input_layer is None:
            raise nnl.errors.InputLayerNotDefined
        else:
            if not self._is_compiled:
                if optimizer is None:
                    raise nnl.errors.OptimizerNotSpecify
                elif loss is None:
                    raise nnl.errors.LossNotSpecify
                else:

                    optimizer_class = nnl.optimizers.get_optimizer(optimizer)
                    if optimizer_class is None:
                        raise nnl.errors.WrongOptimizer(optimizer)
                    else:
                        self._optimizer = optimizer_class()

                    loss_func = nnl.losses.get_loss(loss)
                    if loss_func is None:
                        raise nnl.errors.WrongLoss(loss)
                    else:
                        self._loss = loss_func

                    self._variables = np.zeros(self._get_variables_count())
                    weight_pointer_start = 0
                    prev_node_count = self._input_layer.node_count

                    for i in range(len(self._layers)):
                        curr_node_count = self._layers[i].node_count

                        current_weight = self._weight_initialization(
                            prev_nodes=prev_node_count,
                            curr_nodes=curr_node_count
                        )

                        weight_pointer_end = weight_pointer_start + prev_node_count * curr_node_count

                        self._variables.put(
                            indices=range(weight_pointer_start, weight_pointer_end),
                            values=current_weight
                        )

                        weight_pointer_start = weight_pointer_end + curr_node_count
                        prev_node_count = curr_node_count

                self._is_compiled = True
            else:
                raise nnl.errors.AlreadyCompiled

    def predict(self,
                x: np.ndarray = None,
                verbose: int = 0,
                steps: int = None):

        if self._is_compiled:
            global_time_start = t = time.clock()
            if verbose == 1:
                print('Prediction process started...')
                nnl.print_progress_bar(0, steps, time.clock() - global_time_start)

            if steps is None:
                steps = 0

            if steps > 0:
                input_size = x.shape[0]
                output_size = self._layers[-1].node_count

                watch_size = input_size // steps
                overflow = input_size % steps

                stop_cut = 0
                result = np.zeros((input_size, output_size))

                for i in range(steps):
                    start_cut = stop_cut
                    stop_cut += watch_size + (overflow > 0)

                    if overflow > 0:
                        overflow -= 1

                    piece_result = self.predict_on_batch(x[start_cut:stop_cut])
                    result.put(range(start_cut * output_size, stop_cut * output_size), piece_result)

                    t_end = time.clock()
                    if verbose == 1 and t_end - t > 0.1:
                        t = t_end
                        nnl.print_progress_bar(i, steps, t - global_time_start)
            else:
                result = self.predict_on_batch(x)

            if verbose == 1:
                nnl.print_progress_bar(steps, steps, time.clock() - global_time_start)

            return result
        else:
            raise nnl.errors.NotCompiled

    def predict_on_batch(self, x: np.ndarray):
        if self._is_compiled:
            data = x.copy()
            if len(data.shape) >= 2 and data.shape[1] == self._input_layer.node_count:

                for i in range(len(self._layers)):
                    weight = self._get_weight(layer_number=i)
                    bias = self._get_bias(layer_number=i)
                    multi = np.dot(data, weight) + bias * self._layers[i].bias_flag
                    data = self._layers[i].activation(multi)

                return data
            else:
                raise nnl.errors.WrongInputShape((None, self._input_layer.node_count), data.shape)
        else:
            raise nnl.errors.NotCompiled

    def _loss_wrapper(self, y):
        return lambda x: self._loss(x, y)

    def _feedforward(self, x: np.ndarray):
        if self._is_compiled:
            data = x.copy()
            non_activated = []
            activated = []
            if len(data.shape) >= 2 and data.shape[1] == self._input_layer.node_count:

                for i in range(len(self._layers)):
                    weight = self._get_weight(layer_number=i)
                    bias = self._get_bias(layer_number=i)
                    multi = np.dot(data, weight) + bias * self._layers[i].bias_flag
                    non_activated.append(multi)

                    data = self._layers[i].activation(multi)
                    activated.append(data)

                return non_activated, activated
            else:
                raise nnl.errors.WrongInputShape((None, self._input_layer.node_count), data.shape)
        else:
            raise nnl.errors.NotCompiled

    def fit(self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            batch_size: int = None,
            epochs: int = 1,
            verbose: int = 1,
            shuffle: bool = True):

        if self._is_compiled:
            if batch_size is None:
                batch_size = 32

            in_s = y.shape[0]
            iterations = in_s // batch_size if in_s % batch_size == 0 else in_s // batch_size + 1
            gradient = np.zeros_like(self._variables)

            for epoch in range(epochs):
                epoch_time = time.clock()
                loss_sum = 0

                pfx = f'Epoch {epoch + 1}/{epochs} & Progress:'
                sfx = f'Loss = {loss_sum}'
                if verbose == 1:
                    nnl.print_progress_bar(0, iterations, time.clock() - epoch_time, prefix=pfx, suffix=sfx)

                for k in range(iterations):
                    data = x[batch_size * k:batch_size * (k + 1)]
                    target = y[batch_size * k:batch_size * (k + 1)]

                    non_activated, activated = self._feedforward(data)
                    loss_wrapper = self._loss_wrapper(target)
                    loss_sum += loss_wrapper(activated[-1])

                    delta = nnl.diff(loss_wrapper, activated[-1]) * \
                        nnl.diff(self._layers[-1].activation, non_activated[-1])

                    for i in range(len(self._layers) - 1, 0, -1):
                        weight_size = self._layers[i].node_count * self._layers[i - 1].node_count
                        bias_size = self._layers[i].node_count

                        weight_begin = self._get_variables_count(stop=i)
                        weight_end = weight_begin + weight_size

                        d_bias = np.sum(delta, axis=0)
                        d_weight = np.dot(non_activated[i - 1].T, delta).reshape(weight_size)

                        gradient.put(indices=range(weight_begin, weight_end), values=d_weight)
                        if self._layers[i].bias_flag:
                            gradient.put(indices=range(weight_end, weight_end + bias_size), values=d_bias)

                        next_weight = self._get_weight(layer_number=i)
                        delta = np.dot(delta, next_weight.T) * \
                            nnl.diff(self._layers[i - 1].activation, non_activated[i - 1])

                    weight_size = self._input_layer.node_count * self._layers[0].node_count
                    bias_size = self._input_layer.node_count

                    d_bias = np.sum(delta, axis=0)
                    d_weight = np.dot(data.T, delta).reshape(weight_size)

                    gradient.put(indices=range(0, weight_size), values=d_weight)
                    if self._layers[0].bias_flag:
                        gradient.put(indices=range(weight_size, weight_size + bias_size), values=d_bias)

                    if verbose == 1:
                        sfx = 'Loss = {:0.4f}'.format(round(loss_sum / (k + 1), 4))
                        nnl.print_progress_bar(k + 1, iterations, time.clock() - epoch_time, prefix=pfx, suffix=sfx)

                self._optimizer.optimize(trainable_variables=self._variables, gradient_vector=gradient)
        else:
            raise nnl.errors.NotCompiled

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