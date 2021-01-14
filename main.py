import numpy as np

from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    c = Constructor()

    input_s = 28 * 28
    batch_s = 3

    c.add_input(input_s)
    c.add_layer(300, f.relu)
    c.add_layer(6, f.relu)

    loss_func = f.mse
    optimizer = f.adam
    nn = c.compile(loss_func=loss_func, optimizer=optimizer)

    data = np.random.random((batch_s, input_s)) / input_s

    res = nn.predict(data)
    print(res)


if __name__ == '__main__':
    main()
