def test_derivative():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_der():
        x = np.array([3.5], dtype='float64')
        target = 3 * x ** 2
        value = AutoDiff.derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

    def multi_point_der():
        x = np.array([1, 3.5, 5], dtype='float64')
        target = 3 * x ** 2
        value = AutoDiff.derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_and_func_der():
        x = np.array([1, 1, 4], dtype='float64')
        target = np.array([2 * x[0], 1 / (2 * np.sqrt(x[1])), 1 / x[2]])
        value = AutoDiff.derivative(func=lambda t: np.array([t[0] ** 2, np.sqrt(t[1]), np.log(t[2])]), x=x)

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
        value = AutoDiff.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():
        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = AutoDiff.gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_grad()
    multi_point_grad()


def test_jacobian():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    def single_point_jac():
        x = np.array([[1, 1, 4]], dtype='float64')
        target = - np.repeat(x, x.shape[-1], axis=0).T + np.diag(x[0]) + np.diag((np.sum(x) - x)[0]) / np.sum(x) ** 2
        value = AutoDiff.jacobian(func=lambda t: t / np.sum(t, axis=-1), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert not np.allclose(target, value), error_prompt

    # def multi_point_jac():
    #     def helper(t):
    #         s = np.sum(t, axis=-1).reshape(-1, 1)
    #         tmp = np.repeat(t, t.shape[-1], axis=-2).reshape((t.shape[-2], t.shape[-1], t.shape[-1]))
    #         e = np.array([np.diag(np.ones(t.shape[-1]))])
    #         template = np.repeat(e, t.shape[-2], axis=0)
    #         matrix = - tmp + np.array([t.T]).T * template + np.array([(s - t).T]).T * template
    #         return matrix / s.reshape(-1, 1, 1) ** 2
    #
    #     x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
    #     target = helper(t=x)
    #     value = AutoDiff.jacobian(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)
    #
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target: {target}\n' \
    #                    f'    Value : {value}'
    #
    #     assert np.allclose(target, value), error_prompt

    single_point_jac()
    # multi_point_jac()


def test_jacobian_vector_product():
    import numpy as np
    from nnlibrary.differentiators import Gradient

    gradient = Gradient()

    def single_point_grad():
        x = np.array([1, 1, 4], dtype='float64')
        target = np.array([2 * x[0], 1 / (2 * np.sqrt(x[1])), 1 / x[2]])
        value = gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():
        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = np.array([2 * x[:, 0], 1 / (2 * np.sqrt(x[:, 1])), 1 / x[:, 2]]).T
        value = gradient(func=lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    # single_point_grad()
    # multi_point_grad()
