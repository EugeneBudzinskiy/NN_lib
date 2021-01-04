import numpy as np

from NeuralNetwork import NeuralNetwork


def main():
    nn = NeuralNetwork()

    nn.add_layer(2)
    nn.add_layer(4, 'relu')
    nn.add_layer(8, 'relu')
    nn.add_layer(16, 'relu')

    nn.show_structure()
    nn.compile()

    data = np.random.random(2)

    answer = nn.predict(data)
    print(answer)


if __name__ == '__main__':
    main()
