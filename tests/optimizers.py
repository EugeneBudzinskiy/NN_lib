def test_optimizer_sgd():
    def lr_0_1_mom_0_nesterov_false_single():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0, nesterov=False)
        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.09], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_mom_0_9_nesterov_false_single():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.18], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_mom_0_9_nesterov_true_single():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        x = np.array([1.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.19, -0.2349], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_mom_0_9_nesterov_true_multi():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        x = np.array([1.0, 0.5, -0.1], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.array([first_step, second_step])
        target = np.array([[-0.19, -0.095, 0.019],
                           [-0.2349, -0.11745, 0.02349]], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    lr_0_1_mom_0_nesterov_false_single()
    lr_0_1_mom_0_9_nesterov_false_single()
    lr_0_1_mom_0_9_nesterov_true_single()
    lr_0_1_mom_0_9_nesterov_true_multi()


def test_optimizer_rmsprop():
    def lr_0_1_rho_0_9_mom_0():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.0, centered=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.3162279, -0.22589207], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_rho_0_6_mom_0():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.6, momentum=0.0, centered=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.15811348, -0.12424755], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_rho_0_9_mom_0_8():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.8, centered=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.3162279, -0.4788742], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_rho_0_9_mom_0_centered():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.0, centered=True)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.33333302, -0.25076485], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_rho_0_9_mom_0_multi():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.RMSprop(learning_rate=0.1, rho=0.9, momentum=0.0, centered=False)
        x = np.array([10.0, 1.0, -2.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.array([first_step, second_step])
        target = np.array([[-0.3162279, -0.31622767, 0.31622767],
                           [-0.22589207, -0.1849016, 0.20989692]], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    lr_0_1_rho_0_9_mom_0()
    lr_0_1_rho_0_6_mom_0()
    lr_0_1_rho_0_9_mom_0_8()
    lr_0_1_rho_0_9_mom_0_centered()
    lr_0_1_rho_0_9_mom_0_multi()


def test_optimizer_adam():
    def lr_0_1_beta1_0_9_beta2_0_999():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.09997177], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_beta1_0_5_beta2_0_999():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.Adam(learning_rate=0.1, beta_1=0.5, beta_2=0.999, amsgrad=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.09983158], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_beta1_0_9_beta2_0_7():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.7, amsgrad=False)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.10006142], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_beta1_0_9_beta2_0_999_amsgrad():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=True)
        x = np.array([10.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.concatenate([first_step, second_step], axis=None)
        target = np.array([-0.1, -0.09997177], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    def lr_0_1_beta1_0_9_beta2_0_999_multi():
        import numpy as np
        import nnlibrary as nnl

        optimizer = nnl.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
        x = np.array([10.0, 1.0, -2.0], dtype='float64')  # loss = x ** 2 / 2 => d(loss)/dx = x

        first_step = optimizer(gradient_vector=x)
        second_step = optimizer(gradient_vector=x + first_step)

        value = np.array([first_step, second_step])
        target = np.array([[-0.10000038, -0.09999967, 0.0999999],
                           [-0.09997177, -0.0995872, 0.09983301]], dtype='float64')

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target:\n{target}\n' \
                       f'    Value:\n{value}\n'

        assert np.allclose(target, value), error_prompt

    lr_0_1_beta1_0_9_beta2_0_999()
    lr_0_1_beta1_0_5_beta2_0_999()
    lr_0_1_beta1_0_9_beta2_0_7()
    lr_0_1_beta1_0_9_beta2_0_999_amsgrad()
    lr_0_1_beta1_0_9_beta2_0_999_multi()
