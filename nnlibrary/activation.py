import numpy as np


def linear(x):
    return x.copy()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hard_sigmoid(x):
    z = x.copy()
    z[(z >= -2.5) * (z <= 2.5)] *= 0.2
    z[(z >= -2.5) * (z <= 2.5)] += 0.5
    z[z < -2.5] = 0
    z[z > 2.5] = 1
    return z


def relu(x):
    z = x.copy()
    z[z < 0] = 0
    return z


def tanh(x):
    return np.tanh(x)


def exponential(x):
    return np.exp(x)
