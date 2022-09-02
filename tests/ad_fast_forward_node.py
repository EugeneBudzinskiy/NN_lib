def test_variable_features():
    import numpy as np
    from nnlibrary.auto_diff_fast.forward_mode import special_vars

    def addition():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')
        const = np.array([[1, 5, 7, 1]], dtype='float64')

        var_x = special_vars.Node(values=x, partials=np.ones_like(x))
        l_val, r_val = var_x + const, const + var_x

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x + const, np.ones_like(x), const + const, np.ones_like(x)]
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value :\n{value}'

        assert np.allclose(value, value), error_prompt

    # def multiplication():
    #     x = np.array([[-2, 1, 3, 3]], dtype='float64')
    #     a = np.array([[1, 5, 7, 1]], dtype='float64')
    #     target = np.diag(a.flatten())
    #     value = AutoDiff.forward_mode.jacobian(func=lambda t: t * a, x=x)
    #
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target:\n{target}\n' \
    #                    f'    Value :\n{value}'
    #
    #     assert np.allclose(target, value), error_prompt
    #
    # def matrix_multiplication():
    #     x = np.array([[-2, 5, 3]], dtype='float64')
    #     a = np.array([[-2, 1, 3, 3], [-1, -4, 5, 1], [8, 1, 1, 2]], dtype='float64')
    #     target = a.copy()
    #     value = AutoDiff.forward_mode.jacobian(func=lambda t: t @ a, x=x)
    #
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target:\n{target}\n' \
    #                    f'    Value :\n{value}'
    #
    #     assert np.allclose(target, value), error_prompt
    #
    # def summation():
    #     x = np.array([[-2, 5, 3]], dtype='float64')
    #     target = np.ones_like(x).T
    #     value = AutoDiff.forward_mode.jacobian(func=lambda t: np.sum(t, axis=-1), x=x)
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target:\n{target}\n' \
    #                    f'    Value :\n{value}'
    #
    #     assert np.allclose(target, value), error_prompt
    #
    # def dot_producct():
    #     x = np.array([[-2, 5, 3]], dtype='float64')
    #     a = np.array([[-2, 1, 3, 3], [-1, -4, 5, 1], [8, 1, 1, 2]], dtype='float64')
    #     target = a.copy()
    #
    #     value = AutoDiff.forward_mode.jacobian(func=lambda t: np.dot(t, a), x=x)
    #
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target:\n{target}\n' \
    #                    f'    Value :\n{value}'
    #
    #     assert np.allclose(target, value), error_prompt

    # addition()
    # multiplication()
    # matrix_multiplication()
    # summation()
    # dot_producct()