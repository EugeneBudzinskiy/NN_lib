def test_optimizer_sgd():
    def lr_0_1_mom_0_nesterov_false_single(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0, nesterov=False)

        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x
        value = optimizer(gradient_vector=x)

        target = np.array([-0.1], dtype='float64')

        prompt = 'Optimizer SGD: LR=1, MOM=0, Nesterov=False'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.isclose(target, value), error_prompt

        if flag:
            print(f'Params: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def lr_0_1_mom_0_9_nesterov_false_single(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.18], dtype='float64')

        prompt = 'Optimizer SGD: LR=1, MOM=0, Nesterov=False'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Params: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    lr_0_1_mom_0_nesterov_false_single()
    lr_0_1_mom_0_9_nesterov_false_single()
