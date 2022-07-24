def test_derivative():
    import numpy as np
    from nnlibrary.differentiators import Derivative

    derivative = Derivative()

    def single_point_der(flag: bool = False):
        prompt = 'x ** 3'
        x = np.array([3.5], dtype='float64')
        target = 3 * x ** 2
        value = derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    def multi_point_der(flag: bool = False):
        prompt = 'x ** 3'
        x = np.array([1, 3.5, 5], dtype='float64')
        target = 3 * x ** 2
        value = derivative(func=lambda z: z ** 3, x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    def multi_point_and_func_der(flag: bool = False):
        prompt = '[x[0] ** 2, sqrt(x[1]), ln(x[2])]'
        x = np.array([1, 1, 4], dtype='float64')
        target = (lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(x)
        value = derivative(lambda t: np.array([t[0] ** 2, np.sqrt(t[1]), np.log(t[2])]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    single_point_der()
    multi_point_der()
    multi_point_and_func_der()


def test_gradient():
    import numpy as np
    from nnlibrary.differentiators import Gradient

    gradient = Gradient()

    def single_point_grad(flag: bool = False):
        prompt = 'x[0] ** 2 + sqrt(x[1]) + ln(x[2])'
        x = np.array([[1, 1, 4]], dtype='float64')
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]).T)(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    def multi_point_grad(flag: bool = False):
        prompt = 'x[0] ** 2 + sqrt(x[1]) + ln(x[2])'
        x = np.array([[1, 1, 4], [2, 4, 10]], dtype='float64')
        target = (lambda t: np.array([2 * t[:, 0], 1 / (2 * np.sqrt(t[:, 1])), 1 / t[:, 2]]).T)(x)
        value = gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=x)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    single_point_grad()
    multi_point_grad()
