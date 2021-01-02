import numpy as np

from NeuralNetwork import NeuralNetwork


def main():
    nn = NeuralNetwork()

    nn.add_layer(1)
    nn.add_layer(2, 'relu')
    nn.add_layer(3, 'relu')
    nn.add_layer(4, 'sigmoid')

    nn.show_structure()
    nn.compile()

    data = np.array([.1])

    answer = nn.predict(data)
    print(answer)


if __name__ == '__main__':
    main()
