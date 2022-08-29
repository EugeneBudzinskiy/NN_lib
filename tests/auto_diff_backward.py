def test_gradient():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_grad():
        x = np.array([[1, 1, 4]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = AutoDiff.backward_mode.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():
        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = AutoDiff.forward_mode.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def matrix_multiplication_grad():
        x = np.array([[3, 8, 2], [6, 1, 2]], dtype='float64')
        t = np.array([[9, 4, 1], [6, 2, 8], [7, 5, 3]], dtype='float64')
        target = np.array([[np.sum(t[0]), np.sum(t[1]), np.sum(t[2])],
                           [np.sum(t[0]), np.sum(t[1]), np.sum(t[2])]])
        value = AutoDiff.backward_mode.gradient(func=lambda y: np.array([np.sum(np.dot(y, t))]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def for_loop_reassign():
        from nnlibrary.losses import MeanSquaredError

        def func(t):
            var = t
            for _ in range(3 - 1):
                var = var * t
            return MeanSquaredError().__call__(y_predicted=var, y_target=np.ones_like(t))
        x = np.array([[3, 3, 4]], dtype='float64')
        target = 2 * (x ** 3 - np.ones_like(x)) * 3 * x ** 2 / x.shape[-1]
        value = AutoDiff.backward_mode.gradient(func=func, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def sigmoid_grad():
        from nnlibrary.activations import Sigmoid
        sigmoid = Sigmoid()
        x = np.array([[-3, 3, 4]], dtype='float64')
        target = sigmoid(x) * (1 - sigmoid(x))
        value = AutoDiff.backward_mode.gradient(func=lambda t: np.array([np.sum(sigmoid(t))]), x=x)
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def power_grad():
        p = 3
        x = np.array([[-3, 3, 4]], dtype='float64')
        target = p * x ** (p - 1)
        value = AutoDiff.backward_mode.gradient(func=lambda t: np.array([np.sum(t ** p)]), x=x)
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def exponentiation_grad():
        p = 3
        x = np.array([[-3, 3, 4]], dtype='float64')
        target = p ** x * np.log(abs(p))
        value = AutoDiff.backward_mode.gradient(func=lambda t: np.array([np.sum(p ** t)]), x=x)
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_grad()
    multi_point_grad()
    matrix_multiplication_grad()
    for_loop_reassign()
    sigmoid_grad()
    power_grad()
    exponentiation_grad()
