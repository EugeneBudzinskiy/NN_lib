def test_all():
    import tests

    tests.numpy_wrap.test_array_creation()
    tests.numpy_wrap.test_array_manipulation()
    tests.numpy_wrap.test_linalg()
    tests.numpy_wrap.test_math_funcs()
    tests.numpy_wrap.test_random()
    tests.numpy_wrap.test_typing()

    tests.differentiators.test_derivative()
    tests.differentiators.test_gradient()

    tests.auto_diff_forward.test_derivative()
    tests.auto_diff_forward.test_gradient()
    tests.auto_diff_forward.test_jacobian()
    tests.auto_diff_forward.test_jacobian_vector_product()

    tests.auto_diff_fast_forward.test_derivative()
    tests.auto_diff_fast_forward.test_gradient()
    tests.auto_diff_fast_forward.test_jacobian()
    tests.auto_diff_fast_forward.test_jacobian_vector_product()

    tests.auto_diff_reverse.test_gradient()

    tests.ad_fast_forward_node.test_variable_features()

    tests.sequential.test_predict()
    tests.sequential.test_backpropagation()
    tests.sequential.test_fit()

    tests.losses.test_loss_mse()
    tests.losses.test_loss_cce()

    tests.optimizers.test_optimizer_sgd()
    tests.optimizers.test_optimizer_rmsprop()
    tests.optimizers.test_optimizer_adam()

    print('Passed!')

    # import timeit
    #
    # def old_ed():
    #     def model_3l_235_sigmoid_grad_single():
    #         import numpy as np
    #         import nnlibrary as nnl
    #
    #         np.random.seed(13)
    #
    #         model = nnl.models.SequentialOld()
    #
    #         model.add(layer=nnl.layers.Input(node_count=2))
    #         model.add(layer=nnl.layers.Dense(node_count=100, activation=nnl.activations.Sigmoid()))
    #         model.add(layer=nnl.layers.Dense(node_count=100, activation=nnl.activations.Sigmoid()))
    #         model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))
    #
    #         optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `backpropagation` part
    #         loss = nnl.losses.MeanSquaredError()
    #
    #         w_init = nnl.initializers.UniformZeroOne()
    #         b_init = nnl.initializers.Zeros()
    #
    #         model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)
    #
    #         x = np.array([[1, 1]], dtype='float64')
    #         y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]], dtype='float64')
    #
    #         value = model.backpropagation(x=x, y=y)
    #     return model_3l_235_sigmoid_grad_single()
    #
    # def new_ed():
    #     def model_3l_235_sigmoid_grad_single():
    #         import numpy as np
    #         import nnlibrary as nnl
    #
    #         np.random.seed(13)
    #
    #         model = nnl.models.Sequential()
    #
    #         model.add(layer=nnl.layers.Input(node_count=2))
    #         model.add(layer=nnl.layers.Dense(node_count=100, activation=nnl.activations.Sigmoid()))
    #         model.add(layer=nnl.layers.Dense(node_count=100, activation=nnl.activations.Sigmoid()))
    #         model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))
    #
    #         optimizer = nnl.optimizers.SGD(learning_rate=1)  # Doesn't affect `backpropagation` part
    #         loss = nnl.losses.MeanSquaredError()
    #
    #         w_init = nnl.initializers.UniformZeroOne()
    #         b_init = nnl.initializers.Zeros()
    #
    #         model.compile(optimizer=optimizer, loss=loss, weight_initializer=w_init, bias_initializer=b_init)
    #
    #         x = np.array([[1, 1]], dtype='float64')
    #         y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]], dtype='float64')
    #
    #         value = model.backpropagation(x=x, y=y)
    #     return model_3l_235_sigmoid_grad_single()
    #
    # num = 4
    # time_old = timeit.timeit(lambda: old_ed(), number=num)
    # time_new = timeit.timeit(lambda: new_ed(), number=num)
    # print(f'old: {time_old}')
    # print(f'new: {time_new}')
    # print(f'old / new: {time_old / time_new}')

    # import timeit
    #
    # def old_ed():
    #     a = 2.
    #     b = 3.
    #     c = 5.
    #     return a + b + c
    #
    # def new_ed():
    #     import numpy as np
    #     a = np.array([2.], dtype='float64')
    #     b = np.array([3.], dtype='float64')
    #     c = np.array([5.], dtype='float64')
    #     return float(a[0]) + float(b[0]) + float(c[0])
    #
    # num = int(1e5)
    # time_old = timeit.timeit(lambda: old_ed(), number=num)
    # time_new = timeit.timeit(lambda: new_ed(), number=num)
    # print(f'old: {time_old}')
    # print(f'new: {time_new}')
    # print(f'old / new: {time_old / time_new}')
    # print(f'new / old: {time_new / time_old}')
