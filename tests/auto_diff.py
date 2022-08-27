def test_derivative():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_der():
        x = np.array([3.5], dtype='float64')
        target = 3 * x ** 2
        value = AutoDiff.forward_mode.derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

    def multi_point_der():
        x = np.array([1, 3.5, 5], dtype='float64')
        target = 3 * x ** 2
        value = AutoDiff.forward_mode.derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_and_func_der():
        x = np.array([1, 1, 4], dtype='float64')
        target = np.array([2 * x[0], 1 / (2 * np.sqrt(x[1])), 1 / x[2]])
        value = AutoDiff.forward_mode.derivative(func=lambda t: np.array([t[0] ** 2, np.sqrt(t[1]), np.log(t[2])]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_der()
    multi_point_der()
    multi_point_and_func_der()


def test_gradient():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_grad():
        x = np.array([[1, 1, 4]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = AutoDiff.forward_mode.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():
        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = AutoDiff.forward_mode.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_grad()
    multi_point_grad()


def test_jacobian():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def jac_simple_func():
        def func(t):
            return np.array([[t[0, 0] ** 2 * t[0, 1], t[0, 0] * 5 + np.sin(t[0, 1])]])

        x = np.array([[2, 3]], dtype='float64')
        target = np.array([[2 * x[0, 0] * x[0, 1], x[0, 0] ** 2], [5, np.cos(x[0, 1])]], dtype='float64')
        value = AutoDiff.forward_mode.jacobian(func=func, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def jac_polar_transform():
        def func(t):
            return np.array([[t[0, 0] * np.cos(t[0, 1]), t[0, 0] * np.sin(t[0, 1])]])

        x = np.array([[2, 3]], dtype='float64')
        target = np.array([
            [np.cos(x[0, 1]), - x[0, 0] * np.sin(x[0, 1])],
            [np.sin(x[0, 1]), x[0, 0] * np.cos(x[0, 1])]
        ], dtype='float64')
        value = AutoDiff.forward_mode.jacobian(func=func, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def jac_sphere_transform():
        def func(t):
            return np.array([[
                t[0, 0] * np.sin(t[0, 1]) * np.cos(t[0, 2]),
                t[0, 0] * np.sin(t[0, 1]) * np.sin(t[0, 2]),
                t[0, 0] * np.cos(t[0, 1])
            ]])

        x = np.array([[2, 3, 5]], dtype='float64')
        target = np.array([
            [np.sin(x[0, 1]) * np.cos(x[0, 2]),
             x[0, 0] * np.cos(x[0, 1]) * np.cos(x[0, 2]),
             - x[0, 0] * np.sin(x[0, 1]) * np.sin(x[0, 2])],
            [np.sin(x[0, 1]) * np.sin(x[0, 2]),
             x[0, 0] * np.cos(x[0, 1]) * np.sin(x[0, 2]),
             x[0, 0] * np.sin(x[0, 1]) * np.cos(x[0, 2])],
            [np.cos(x[0, 1]), - x[0, 0] * np.sin(x[0, 1]), 0]
        ], dtype='float64')
        value = AutoDiff.forward_mode.jacobian(func=func, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def jac_non_square():
        def func(t):
            x1, x2, x3 = t[0, 0], t[0, 1], t[0, 2]
            return np.array([[x1, x3 * 5, x2 ** 2 * 4 - x3 * 2, x3 * np.sin(x1)]])

        x = np.array([[2, 3, 5]], dtype='float64')
        target = np.array([
            [1., 0., 0.],
            [0., 0., 5.],
            [0., 8. * x[0, 1], - 2.],
            [x[0, 2] * np.cos(x[0, 0]), 0, np.sin(x[0, 0])]
        ], dtype='float64')
        value = AutoDiff.forward_mode.jacobian(func=func, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_jac():
        def jacobian_helper(t):
            s = np.sum(t, axis=-1).reshape(-1, 1)
            tmp = np.repeat(t, t.shape[-1], axis=-2).reshape((t.shape[-2], t.shape[-1], t.shape[-1]))
            template = np.repeat(np.array([np.diag(np.ones(t.shape[-1]))]), t.shape[-2], axis=0)
            matrix = - tmp + np.array([t.T]).T * template + np.array([(s - t).T]).T * template
            return matrix / s.reshape(-1, 1, 1) ** 2

        x = np.array([[1, 3, 4], [2, 4, 10]], dtype='float64')
        target = jacobian_helper(t=x)
        value = AutoDiff.forward_mode.jacobian(func=lambda t: t / np.sum(t, axis=-1).reshape(-1, 1), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    jac_simple_func()
    jac_polar_transform()
    jac_sphere_transform()
    jac_non_square()
    # multi_point_jac()


def test_jacobian_vector_product():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_jvp():
        def jvp_helper(t, vec):
            s = np.sum(t, axis=-1).reshape(-1, 1)
            tmp = np.repeat(t, t.shape[-1], axis=-2).reshape((t.shape[-2], t.shape[-1], t.shape[-1]))
            template = np.repeat(np.array([np.diag(np.ones(t.shape[-1]))]), t.shape[-2], axis=0)
            matrix = - tmp + np.array([t.T]).T * template + np.array([(s - t).T]).T * template
            return np.array([np.dot(vec[i], matrix[i]) for i in range(t.shape[0])]) / s ** 2

        x = np.array([[1, 5, 7]], dtype='float64')
        v = np.array([[3, 1, 2]], dtype='float64')
        target = jvp_helper(t=x, vec=v)
        value = AutoDiff.forward_mode.jvp(func=lambda t: t / np.sum(t, axis=-1).reshape(-1, 1), x=x, vector=v)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_jvp():
        def jvp_helper(t, v):
            s = np.sum(t, axis=-1).reshape(-1, 1)
            tmp = np.repeat(t, t.shape[-1], axis=-2).reshape((t.shape[-2], t.shape[-1], t.shape[-1]))
            template = np.repeat(np.array([np.diag(np.ones(t.shape[-1]))]), t.shape[-2], axis=0)
            matrix = - tmp + np.array([t.T]).T * template + np.array([(s - t).T]).T * template
            return np.array([np.dot(v[i], matrix[i]) for i in range(t.shape[0])]) / s ** 2

        x = np.array([[1, 5, 7], [2, 4, 10]], dtype='float64')
        v = np.array([[3, 1, 2], [8, 12, 2]], dtype='float64')
        target = jvp_helper(t=x, v=v)
        value = AutoDiff.forward_mode.jvp(func=lambda t: t / np.sum(t, axis=-1).reshape(-1, 1), x=x, vector=v)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(target, value), error_prompt

    # single_point_jvp()
    # multi_point_jvp()
