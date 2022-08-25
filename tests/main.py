def test_all():
    import tests

    tests.differentiators.test_derivative()
    tests.differentiators.test_gradient()

    tests.autodiff.test_derivative()
    tests.autodiff.test_gradient()
    tests.autodiff.test_jacobian()
    tests.autodiff.test_jacobian_vector_product()

    tests.sequential.test_predict()
    tests.sequential.test_backpropagation()
    tests.sequential.test_fit()

    tests.losses.test_loss_mse()
    tests.losses.test_loss_cce()

    tests.optimizers.test_optimizer_sgd()
    tests.optimizers.test_optimizer_rmsprop()
    tests.optimizers.test_optimizer_adam()

    print('Passed!')
