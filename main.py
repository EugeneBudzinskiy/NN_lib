import numpy as np

from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    c = Constructor()

    c.add_input(2)
    c.add_layer(3, f.sigmoid)
    c.add_layer(3, f.sigmoid)

    loss_func = f.mse
    optimizer = f.adam
    nn = c.compile(loss_func=loss_func, optimizer=optimizer)

    data = np.array([[1, 0]])
    target = np.array([[0, 1, 0]])

    for _ in range(1000):
        nn.learn(data, target)


if __name__ == '__main__':
    main()
