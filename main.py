from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    c = Constructor()

    c.add_input(784)
    c.add_layer(300, f.relu)
    c.add_layer(10, f.relu)

    loss_func = f.mse
    optimizer = f.adam
    nn = c.compile(loss_func=loss_func, optimizer=optimizer)
    print(nn.learning_rate)


if __name__ == '__main__':
    main()
