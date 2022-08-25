def test_derivative():
    import numpy as np
    from nnlibrary.auto_diff import AutoDiff

    x = np.array([[1, 2, 3], [3, 4, 5]], dtype='float64')
    v = np.array([[1, 2, 3], [3, 4, 5]], dtype='float64')
    func = lambda t: np.square(t) + t * t * t

    value = AutoDiff.jvp(func=func, x=x, vector=v)
    target = np.array([np.diag((x * (3 * x + 2))[0]) @ v[0], np.diag((x * (3 * x + 2))[1]) @ v[1]])

    assert np.allclose(value, target), ''


