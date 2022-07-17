import numpy as np

import nnlibrary as nnl

np.random.seed(13)


def main():
    model = nnl.models.Sequential()

    model.add(layer=nnl.layers.Input(node_count=2))
    model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Linear()))
    model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Linear()))

    optimizer = nnl.optimizers.SGD(learning_rate=1)
    loss = nnl.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)

    # res = model.predict(x=np.array([1, 1]))
    # print(res)

    model.backpropagation(x=np.array([1, 1]), y=np.array([0.1, 0.1, 0.1, 0.1, 0.1]))

    print(model.trainable_variables.get_all())

    model.backpropagation(x=np.array([1, 1]), y=np.array([0.1, 0.1, 0.1, 0.1, 0.1]))


if __name__ == '__main__':
    main()
