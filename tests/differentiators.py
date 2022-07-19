def test_derivative():
    import numpy as np
    from nnlibrary.differentiators import Derivative

    derivative = Derivative()

    def single_point_der():
        x = np.array([3.5], dtype='float64')
        print(f'Function: x ** 3\n'
              f'  Point = {x}\n'
              f'  Desire Output: {3 * x ** 2}\n'
              f'  Real Output:   {derivative(func=lambda z: z ** 3, x=x)}\n')

    def multi_point_der():
        x = np.array([1, 3.5, 5], dtype='float64')
        print(f'Function: x ** 3\n'
              f'  Point = {x}\n'
              f'  Desire Output: {3 * x ** 2}\n'
              f'  Real Output:   {derivative(func=lambda z: z ** 3, x=x)}\n')

    def multi_point_and_func_der():
        x = np.array([1, 1, 4], dtype='float64')
        print(f'Function: [x[0] ** 2, sqrt(x[1]), ln(x[2])]\n'
              f'  Point = {x}\n'
              f'  Desire Output: {(lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(x)}\n'
              f'  Real Output:   {derivative(lambda t: np.array([t[0] ** 2, np.sqrt(t[1]), np.log(t[2])]), x=x)}\n')
    print('-== Derivative Tests ==-\n')
    single_point_der()
    multi_point_der()
    multi_point_and_func_der()
    print('=' * 32, '\n')


def test_gradient():
    import numpy as np
    from nnlibrary.differentiators import Gradient

    gradient = Gradient()

    print(gradient(lambda t: t[:, 0] ** 2 + np.sqrt(t[:, 1]) + np.log(t[:, 2]), x=np.array([[1, 1, 4]])))
    print((lambda t: np.array([2 * t[0], 1 / (2 * np.sqrt(t[1])), 1 / t[2]]))(np.array([1, 1, 4])))
