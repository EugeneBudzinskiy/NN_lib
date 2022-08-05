def test_loss_mse():
    import numpy as np
    from nnlibrary.losses import MeanSquaredError

    loss = MeanSquaredError()

    def single_point_loss(flag: bool = False):
        prompt = 'Mean Squared Error'
        y_predicted = np.array([3.5, 1.2, 3.3], dtype='float64')
        y_target = np.array([3.0, 2.0, 1.0], dtype='float64')
        target = np.array([2.06], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.isclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  y_predicted = {y_predicted}\n'
                  f'  y_target = {y_target}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    def multi_point_loss(flag: bool = False):
        prompt = 'Mean Squared Error'
        y_predicted = np.array([[3.5, 1.2, 3.3], [0.6, 0.7, 0.0]], dtype='float64')
        y_target = np.array([[3.0, 2.0, 1.0], [1.0, 0.5, 0.2]], dtype='float64')
        target = np.array([1.07], dtype='float64')
        value = loss(y_predicted=y_predicted, y_target=y_target)

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Function: {prompt}\n'
                  f'  y_predicted = {y_predicted}\n'
                  f'  y_target = {y_target}\n'
                  f'  Desire Output: {target}\n'
                  f'  Real Output:   {value}\n')

    single_point_loss()
    multi_point_loss()
