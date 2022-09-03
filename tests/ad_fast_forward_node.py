def test_variable_features():
    import numpy as np
    from nnlibrary.auto_diff_fast.forward_mode.special_vars import Node

    def addition():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')
        const = np.array([[1, 5, 7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(var_x + const, verbose=False)
        r_val = Node.unwrap_if_needed(const + var_x, verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x + const, np.ones_like(x), const + x, np.ones_like(x)]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def subtraction():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')
        const = np.array([[1, 5, 7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(var_x - const, verbose=False)
        r_val = Node.unwrap_if_needed(const - var_x, verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x - const, np.ones_like(x), const - x, - np.ones_like(x)]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def multiplication():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')
        const = np.array([[1, 5, 7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(var_x * const, verbose=False)
        r_val = Node.unwrap_if_needed(const * var_x, verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x * const, const, const * x, const]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def division():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')
        const = np.array([[1, 5, 7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(var_x / const, verbose=False)
        r_val = Node.unwrap_if_needed(const / var_x, verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x / const, 1 / const, const / x, -const / x ** 2]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def matrix_multiplication():
        x = np.array([[-2, 1], [3, 3]], dtype='float64')
        const = np.array([[1, 5], [7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(var_x @ const, verbose=False)
        r_val = Node.unwrap_if_needed(const @ var_x, verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x @ const, np.ones_like(x) @ const, const @ x, const @ np.ones_like(x)]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def summation():
        x = np.array([[-2, 1, 3, 3]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        val = Node.unwrap_if_needed(np.sum(var_x), verbose=False)

        value = [val.values, val.partials]
        target = [np.sum(x), np.sum(np.ones_like(x))]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    def dot_product():
        x = np.array([[-2, 1], [3, 3]], dtype='float64')
        const = np.array([[1, 5], [7, 1]], dtype='float64')

        var_x = Node(values=x, partials=np.ones_like(x))
        l_val = Node.unwrap_if_needed(np.dot(var_x, const), verbose=False)
        r_val = Node.unwrap_if_needed(np.dot(const, var_x), verbose=False)

        value = [l_val.values, l_val.partials, r_val.values, r_val.partials]
        target = [x @ const, np.ones_like(x) @ const, const @ x, const @ np.ones_like(x)]

        error_prompt = f'\n  Target:\n{target}\n  Value :\n{value}'
        assert np.allclose(target, value), error_prompt

    addition()
    subtraction()
    multiplication()
    division()
    matrix_multiplication()
    summation()
    dot_product()
