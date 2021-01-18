import numpy as np

from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    c = Constructor()

    c.add_input(784)
    c.add_layer(300, f.sigmoid)
    c.add_layer(10, f.sigmoid)

    loss_func = f.mse
    optimizer = f.adam
    nn = c.compile(loss_func=loss_func, optimizer=optimizer)

    data = np.random.random((2, 784))
    result = nn.predict(data)

    print(result)


if __name__ == '__main__':
    main()
