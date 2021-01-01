from NeuralNetwork import NeuralNetwork


def main():
    nn = NeuralNetwork()

    nn.add_layer(1, 'linear')
    nn.add_layer(2, 'relu')
    nn.add_layer(3, 'relu')
    nn.add_layer(4, 'sigmoid')

    nn.show_structure()
    nn.compile()

    print(nn.functions)


if __name__ == '__main__':
    main()
