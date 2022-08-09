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

    def lr_0_1_mom_0_9_nesterov_true_single(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.19, -0.2349], dtype='float64')

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

    def lr_0_1_mom_0_9_nesterov_true_multi(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        x = np.array([1.0, 0.5, -0.1], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.array([first_step, second_step])
        target = np.array([[-0.19,  -0.095,  0.019],
                           [-0.2349,  -0.11745,  0.02349]], dtype='float64')

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
    lr_0_1_mom_0_9_nesterov_true_single()
    lr_0_1_mom_0_9_nesterov_true_multi()


def test_optimizer_rmsprop():
    def lr_0_1_rho_0_9_mom_0(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.0)

        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x
        value = optimizer(gradient_vector=x)

        target = np.array([-0.3162279], dtype='float64')

        prompt = 'Optimizer RMSprop: LR=0.1, RHO=0.9, MOM=0'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.isclose(target, value), error_prompt

        if flag:
            print(f'Params: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def lr_0_1_rho_0_6_mom_0(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.6, momentum=0.0)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.15811348, -0.12424755], dtype='float64')

        prompt = 'Optimizer RMSprop: LR=0.1, RHO=0.6, MOM=0'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Params: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    def lr_0_1_rho_0_9_mom_0_8(flag: bool = False):
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.8)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.3162279, -0.4788742], dtype='float64')

        prompt = 'Optimizer RMSprop: LR=0.1, RHO=0.9, MOM=0.8'
        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

        if flag:
            print(f'Params: {prompt}\n'
                  f'  Point = {x}\n'
                  f'  Desire Output:\n{target}\n'
                  f'  Real Output:\n{value}\n')

    # def lr_0_1_rho_0_9_mom_0_centered(flag: bool = False):
    #     import numpy as np
    #     import nnlibrary as nnl
    #
    #     optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.0, centered=True)
    #     x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x
    #
    #     first_step = optimizer(gradient_vector=x)
    #     second_step = optimizer(gradient_vector=x + first_step)
    #
    #     value = np.concatenate([first_step, second_step], axis=None)
    #     target = np.array([-0.33333302, -0.25076485], dtype='float64')
    #
    #     prompt = 'Optimizer RMSprop: LR=0.1, RHO=0.9, MOM=0.8'
    #     error_prompt = f'\n  Target and Value are not the same: \n' \
    #                    f'    Target:\n{target}\n' \
    #                    f'    Value:\n{value}\n'
    #
    #     assert np.allclose(target, value), error_prompt
    #
    #     if flag:
    #         print(f'Params: {prompt}\n'
    #               f'  Point = {x}\n'
    #               f'  Desire Output:\n{target}\n'
    #               f'  Real Output:\n{value}\n')

    lr_0_1_rho_0_9_mom_0()
    lr_0_1_rho_0_6_mom_0()
    lr_0_1_rho_0_9_mom_0_8()
