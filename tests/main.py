def test_all():
    import tests

    tests.differentiators.test_derivative()
    tests.differentiators.test_gradient()

    tests.sequential.test_predict()
    tests.sequential.test_backpropagation()

    tests.losses.test_loss_mse()

    print('Passed!')
