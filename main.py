from NN_Constructor import Constructor
from Functions import Func


def main():
    f = Func()
    constructor = Constructor()

    constructor.add_input(1)
    constructor.add_layer(2, f.relu)
    constructor.add_layer(3, f.relu)
    constructor.add_layer(4, f.relu)

    constructor.show_structure()


if __name__ == '__main__':
    main()
