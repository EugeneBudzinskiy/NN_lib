def test_all():
    import tests

    tests.differentiators.test_derivative()
    tests.differentiators.test_gradient()

    tests.auto_diff.test_derivative_forward()
    tests.auto_diff.test_gradient_forward()
    tests.auto_diff.test_jacobian_forward()
    tests.auto_diff.test_jacobian_vector_product_forward()

    tests.sequential.test_predict()
    tests.sequential.test_backpropagation()
    tests.sequential.test_fit()

    tests.losses.test_loss_mse()
    tests.losses.test_loss_cce()

    tests.optimizers.test_optimizer_sgd()
    tests.optimizers.test_optimizer_rmsprop()
    tests.optimizers.test_optimizer_adam()

    print('Passed!')
