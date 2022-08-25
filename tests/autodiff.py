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
        target = (lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(x)
        value = AutoDiff.derivative(lambda t: np.array([t[0] ** 2, np.sqrt(t[1]), np.log(t[2])]), x=x)

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
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]))(x)
        value = AutoDiff.gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():

        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]).T)(x)
        value = AutoDiff.gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_grad()
    # multi_point_grad()


def test_jacobian():
    import numpy as np
    from nnlibrary.differentiators import Gradient

    gradient = Gradient()

    def single_point_grad():
        x = np.array([1, 1, 4], dtype='float64')
        target = (lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():

        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]).T)(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    # single_point_grad()
    # multi_point_grad()


def test_jacobian_vector_product():
    import numpy as np
    from nnlibrary.differentiators import Gradient

    gradient = Gradient()

    def single_point_grad():
        x = np.array([1, 1, 4], dtype='float64')
        target = (lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_grad():

        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]).T)(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    # single_point_grad()
    # multi_point_grad()
