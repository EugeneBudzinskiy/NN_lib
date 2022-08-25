def test_loss_mse():
    import numpy as np
    from nnlibrary.losses import MeanSquaredError

    loss = MeanSquaredError()

    def single_point_loss():
        y_predicted = np.array([[3.5, 1.2, 3.3]], dtype='float64')
        y_target = np.array([[3.0, 2.0, 1.0]], dtype='float64')
        target = np.array([2.06], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

    def multi_point_loss():
        y_predicted = np.array([[3.5, 1.2, 3.3], [0.6, 0.7, 0.0]], dtype='float64')
        y_target = np.array([[3.0, 2.0, 1.0], [1.0, 0.5, 0.2]], dtype='float64')
        target = np.array([1.07], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_loss()
    multi_point_loss()


def test_loss_cce():
    def single_point_loss():
        import numpy as np
        from nnlibrary.losses import CategoricalCrossentropy

        loss = CategoricalCrossentropy()

        y_predicted = np.array([[0.05, 0.95, 0]], dtype='float64')
        y_target = np.array([[0.0, 1.0, 0.0]], dtype='float64')
        target = np.array([0.051293306], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

    def multi_point_loss():
        import numpy as np
        from nnlibrary.losses import CategoricalCrossentropy

        loss = CategoricalCrossentropy()

        y_predicted = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype='float64')
        y_target = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype='float64')
        target = np.array([1.1769392], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    def multi_point_loss_logits():
        import numpy as np
        from nnlibrary.losses import CategoricalCrossentropy

        loss = CategoricalCrossentropy(from_logits=True)

        y_predicted = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype='float64')
        y_target = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype='float64')

        target = np.array([0.9868950481037163], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

    single_point_loss()
    multi_point_loss()
    multi_point_loss_logits()
