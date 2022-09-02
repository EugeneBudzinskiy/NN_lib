def test_array_creation():
    import numpy as np
    import nnlibrary.numpy_wrap as npw

    def test_empty():
        value = npw.empty((10, 4)).shape
        target = np.empty((10, 4)).shape
        assert np.allclose(target, value), f'Target:\n{target}\nValue :\n{value}'

    def test_empty_like():
        value = npw.empty_like((10, 4)).shape
        target = np.empty((10, 4)).shape
        assert np.allclose(target, value), f'Target:\n{target}\nValue :\n{value}'

    test_empty()


def test_array_manipulation():
    pass


def test_linalg():
    pass


def test_math_funcs():
    pass


def test_random():
    pass


def test_typing():
    pass
