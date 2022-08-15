def test_predict():
    def model_3l_235_sigmoid_weight():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `predict` part
        loss = nnl.losses.MeanSquaredError()  # Doesn't affect `predict` part

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        value = model.trainable_variables.get_all()
        target = np.array([0.77770241, 0.23754122, 0.82427853, 0.9657492, 0.97260111, 0.45344925,
                           0., 0., 0., 0.60904246, 0.77552651, 0.64161334,
                           0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
                           0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
                           0., 0., 0., 0., 0.])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_single():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `predict` part
        loss = nnl.losses.MeanSquaredError()  # Doesn't affect `predict` part

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([1, 1], dtype='float64')

        value = model.predict(x=x)
        target = np.array([[0.72085387, 0.72650737, 0.7709476, 0.7653046, 0.785123]])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_multi():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `predict` part
        loss = nnl.losses.MeanSquaredError()  # Doesn't affect `predict` part

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([[1, 1], [0, 0.5]], dtype='float64')

        value = model.predict(x=x)
        target = np.array([[0.72085387, 0.72650737, 0.7709476, 0.7653046, 0.785123],
                           [0.6690879, 0.6702348, 0.7176329, 0.7061567, 0.72527224]])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_single()
    model_3l_235_sigmoid_multi()


def test_backpropagation():
    def model_3l_235_sigmoid_weight():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        value = model.trainable_variables.get_all()
        target = np.array([0.77770241, 0.23754122, 0.82427853, 0.9657492, 0.97260111, 0.45344925,
                           0., 0., 0., 0.60904246, 0.77552651, 0.64161334,
                           0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
                           0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
                           0., 0., 0., 0., 0.])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_grad_single():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `backpropagation` part
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([1, 1], dtype='float64')
        y = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        target = np.array([0.01718086, 0.01905474, 0.01560694, 0.01718086, 0.01905471, 0.01560694,
                           0.01718085, 0.01905474, 0.01560693, 0.04253256, 0.04238039, 0.04033685,
                           0.04068297, 0.03935043, 0.03849477, 0.03835706, 0.03650749, 0.03682077,
                           0.03561473, 0.03908136, 0.03894156, 0.0370638, 0.03738186, 0.03615743,
                           0.04997217, 0.0497934, 0.04739236, 0.04779906, 0.04623344])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_grad_multi():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `backpropagation` part
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([[1, 1], [0, 1]], dtype='float64')
        y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5]], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        target = np.array([0.0085904, 0.00952737, 0.00780344, 0.01344478, 0.01365507, 0.01189515,
                           0.0134448, 0.01365506, 0.01189514, 0.027215, 0.02714866, 0.02694768,
                           0.02695358, 0.02646597, 0.02520731, 0.02514825, 0.02504581, 0.02503496,
                           0.02461094, 0.02456282, 0.02450117, 0.02425525, 0.02427313, 0.02381176,
                           0.03319946, 0.03312356, 0.03305633, 0.03302884, 0.0324927])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_linear_cce_single():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Linear()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Linear()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `backpropagation` part
        loss = nnl.losses.CategoricalCrossentropy()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([1, 1], dtype='float64')
        y = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        target = np.array([0.07098395, -0.05505513, -0.04471427, 0.07098395, -0.05505514, -0.04471427,
                           0.07098398, -0.05505513, -0.04471427, 0.82251304, -0.17348409, -0.17348412,
                           -0.17348409, -0.17348412, 0.57091224, -0.12041658, -0.12041658, -0.12041658,
                           -0.12041658, 0.6027972, -0.12714174, -0.12714174, -0.12714174, -0.12714171,
                           0.4717728, -0.09950612, -0.09950612, -0.09950612, -0.09950612])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        output = model.predict(x=x)
        print(model.loss(y_target=y, y_predicted=output))
        exit()

        assert np.allclose(target, value), error_prompt

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_grad_single()
    model_3l_235_sigmoid_grad_multi()
    model_3l_235_linear_cce_single()


def test_fit():
    def model_3l_235_sigmoid_weight():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        value = model.trainable_variables.get_all()
        target = np.array([0.77770241, 0.23754122, 0.82427853, 0.9657492, 0.97260111, 0.45344925,
                           0., 0., 0., 0.60904246, 0.77552651, 0.64161334,
                           0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
                           0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
                           0., 0., 0., 0., 0.])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_cost_single():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([1, 1], dtype='float64')
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype='float64')

        before_fit = model.loss(y_predicted=model.predict(x=x), y_target=y)  # Save Loss before Fit

        model.fit(x=x, y=y, epochs=1, batch_size=2, shuffle=False)
        target = np.array([0.06128797, 0.056028955])  # targeted [`before`, `after`] fit values

        after_fit = model.loss(y_predicted=model.predict(x=x), y_target=y)  # Save Loss after Fit
        value = np.concatenate([before_fit, after_fit], axis=None)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n' \
                       f'       Before fit: {target[0]}\n' \
                       f'        After fit: {target[1]}\n' \
                       f'    Value:\n' \
                       f'       Before fit: {value[0]}\n' \
                       f'        After fit: {value[1]}\n'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_cost_multi():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0],
            [.5, .5]
        ], dtype='float64')
        y = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1]
        ], dtype='float64')

        before_fit = model.loss(y_predicted=model.predict(x=x), y_target=y)  # Save Loss before Fit

        model.fit(x=x, y=y, epochs=1, batch_size=2, shuffle=False)
        target = np.array([0.19699292, 0.1736497])  # targeted [`before`, `after`] fit values

        after_fit = model.loss(y_predicted=model.predict(x=x), y_target=y)  # Save Loss after Fit
        value = np.concatenate([before_fit, after_fit], axis=None)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n' \
                       f'       Before fit: {target[0]}\n' \
                       f'        After fit: {target[1]}\n' \
                       f'    Value:\n' \
                       f'       Before fit: {value[0]}\n' \
                       f'        After fit: {value[1]}\n'

        assert np.allclose(target, value), error_prompt

    def model_3l_235_sigmoid_cost_10_epochs():
        import numpy as np
        import nnlibrary as nnl

        np.random.seed(13)

        model = nnl.models.Sequential()

        model.add(layer=nnl.layers.Input(node_count=2))
        model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
        model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

        optimizer = nnl.optimizers.SGD(learning_rate=1)
        loss = nnl.losses.MeanSquaredError()

        w_init = nnl.variables.UniformZeroOne()
        b_init = nnl.variables.Zeros()

        model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)

        x = np.array([
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0],
            [.5, .5]
        ], dtype='float64')
        y = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1]
        ], dtype='float64')

        before_fit = np.mean(model.loss(y_predicted=model.predict(x=x), y_target=y))  # Save Loss before Fit

        model.fit(x=x, y=y, epochs=10, batch_size=2, shuffle=False)
        target = np.array([0.19699292, 0.11798767])  # targeted [`before`, `after`] fit values

        after_fit = np.mean(model.loss(y_predicted=model.predict(x=x), y_target=y))  # Save Loss after Fit
        value = np.array([before_fit, after_fit])

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n' \
                       f'       Before fit: {target[0]}\n' \
                       f'        After fit: {target[1]}\n' \
                       f'    Value:\n' \
                       f'       Before fit: {value[0]}\n' \
                       f'        After fit: {value[1]}\n'

        assert np.allclose(target, value), error_prompt

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_cost_single()
    model_3l_235_sigmoid_cost_multi()
    model_3l_235_sigmoid_cost_10_epochs()
