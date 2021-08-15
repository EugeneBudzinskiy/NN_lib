def diff(func, x, epsilon: float = 1e-5):
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)
