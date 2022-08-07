def test_optimizer_sgd():
    def learning_rate_1_momentum_0_nesterov_false(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1, momentum=0, nesterov=False)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([1, 1], dtype='float64')
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype='float64')

        before_fit = model.trainable_variables.get_all()  # Save Weights before Fit

        model.fit(x=x, y=y)
        target = np.array([0.06128797, 0.056028955])  # targeted [`before`, `after`] fit values

        after_fit = np.mean(model.loss(y_predicted=model.predict(x=x), y_target=y))  # Save Weights after Fit
        value = np.array([before_fit, after_fit])

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n' \
                       f'       Before fit: {target[0]}\n' \
                       f'        After fit: {target[1]}\n' \
                       f'    Value:\n' \
                       f'       Before fit: {value[0]}\n' \
                       f'        After fit: {value[1]}\n'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    learning_rate_1_momentum_0_nesterov_false()
