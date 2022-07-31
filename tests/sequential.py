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
                           0.,         0.,         0.,         0.60904246, 0.77552651, 0.64161334,
                           0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
                           0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
                           0.,         0.,         0.,         0.,         0.])

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
    # import numpy as np
    # import nnlibrary as nnl
    #
    # np.random.seed(13)
    #
    # model = nnl.models.Sequential()
    #
    # model.add(layer=nnl.layers.Input(node_count=2))
    # model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
    # model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))
    #
    # optimizer = nnl.optimizers.SGD(learning_rate=1)
    # loss = nnl.losses.MeanSquaredError()
    #
    # model.compile(optimizer=optimizer, loss=loss)
    #
    # x = np.array([[1, 1], [1, 1]])
    # y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])
    #
    # print(model.backpropagation(x=x, y=y))
    #
    # print(model.trainable_variables.get_all())
    #
    # model.backpropagation(x=x, y=y)
    # exit()

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
        y = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype='float64')

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

        x = np.array([[1, 1]], dtype='float64')
        y = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype='float64')

        value = model.backpropagation(x=x, y=y)
        # print(value)
        # exit()  # TODO fix backprop!
        #
        # target = np.array([0.77770241, 0.23754122, 0.82427853, 0.9657492, 0.97260111, 0.45344925,
        #                    0.,         0.,         0.,         0.60904246, 0.77552651, 0.64161334,
        #                    0.72201823, 0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
        #                    0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378, 0.94909418,
        #                    0., 0., 0., 0., 0.])
        #
        # prompt = '3 Layers: (2 Input - 3 Sigmoid - 5 Sigmoid)'
        # error_prompt = f'\n  Target and Value are not the same: \n' \
        #                f'    Target:\n{target}\n' \
        #                f'    Value:\n{value}'
        #
        # # assert np.allclose(target, value), error_prompt
        #
        # if flag:
        #     print(f'Structure: {prompt}\n'
        #           f'  Point = {x}\n'
        #           f'  Desire Output:\n{target}\n'
        #           f'  Real Output:\n{value}\n')

    model_3l_235_sigmoid_weight()
    model_3l_235_sigmoid_single()
