import numpy as np

import nnlibrary as nnl

np.random.seed(13)


def main():
    model = nnl.models.Sequential()

    model.add(layer=nnl.layers.Input(node_count=2))
    model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
    model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

    optimizer = nnl.optimizers.SGD(learning_rate=1)
    loss = nnl.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)

    x = np.array([[1, 1], [1, 1]])
    y = np.array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]])

    # res = model.predict(x=x)
    # print(res)

    print(model.backpropagation(x=x, y=y))

    # print(model.trainable_variables.get_all())

    # model.backpropagation(x=x, y=y)


if __name__ == '__main__':
    main()
