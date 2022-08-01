def test_predict():
    def model_3l_235_sigmoid_weight(flag: bool = False):
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

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Distribution: Weights = Uniform(0, 1), Biases = 0\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def model_3l_235_sigmoid_single(flag: bool = False):
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

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def model_3l_235_sigmoid_multi(flag: bool = False):
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

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_single()
    model_3l_235_sigmoid_multi()


def test_backpropagation():
    def model_3l_235_sigmoid_weight(flag: bool = False):
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

        value = model.trainable_variables.get_all()
        target = np.array([0.77770241, 0.23754122, 0.82427853, 0.9657492, 0.97260111, 0.45344925,
                           0., 0., 0., 0.60904246, 0.77552651, 0.64161334,
                           0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
                           0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
                           0., 0., 0., 0., 0.])

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def model_3l_235_sigmoid_single(flag: bool = False):
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
        y = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        target = np.array([0.01718086, 0.01905474, 0.01560694, 0.01718086, 0.01905471, 0.01560694,
                           0.01718085, 0.01905474, 0.01560693, 0.04253256, 0.04238039, 0.04033685,
                           0.04068297, 0.03935043, 0.03849477, 0.03835706, 0.03650749, 0.03682077,
                           0.03561473, 0.03908136, 0.03894156, 0.0370638, 0.03738186, 0.03615743,
                           0.04997217, 0.0497934, 0.04739236, 0.04779906, 0.04623344])

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def model_3l_235_sigmoid_multi(flag: bool = False):
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

        x = np.array([[1, 1], [0, 1]], dtype='float64')
        y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5]], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        target = np.array([0.0085904, 0.00952737, 0.00780344, 0.01344478, 0.01365507, 0.01189515,
                           0.0134448, 0.01365506, 0.01189514, 0.027215, 0.02714866, 0.02694768,
                           0.02695358, 0.02646597, 0.02520731, 0.02514825, 0.02504581, 0.02503496,
                           0.02461094, 0.02456282, 0.02450117, 0.02425525, 0.02427313, 0.02381176,
                           0.03319946, 0.03312356, 0.03305633, 0.03302884, 0.0324927])

        prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Structure: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_single()
    model_3l_235_sigmoid_multi()
