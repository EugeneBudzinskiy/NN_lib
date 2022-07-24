def test_all():
    import tests

    tests.differentiators.test_derivative()
    tests.differentiators.test_gradient()

    tests.sequential.test_predict()

    print('Passed!')
