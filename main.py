from nnlibrary import NNLib


def main():
    nnl = NNLib()

    nnl.constructor.add(nnl.layers.Input(node_count=3))
    nnl.constructor.add(nnl.layers.Dense(node_count=5, activation=nnl.activation.Sigmoid))
    nnl.constructor.show_structure()


if __name__ == '__main__':
    main()
