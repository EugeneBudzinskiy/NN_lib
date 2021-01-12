from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    c = Constructor()

    c.add_input(1)
    c.add_layer(2, f.relu)
    c.add_layer(3, f.relu)
    c.add_layer(4, f.relu)

    loss_func = f.mse
    optimizer = f.adam
    nn = c.compile(loss_func, optimizer)

    nn.predict(1)


if __name__ == '__main__':
    main()
