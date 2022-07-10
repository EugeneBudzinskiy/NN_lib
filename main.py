import numpy as np

import nnlibrary as nnl

np.random.seed(13)


def main():
    model = nnl.models.Sequential()

    model.add(layer=nnl.layers.Input(node_count=2))
    model.add(layer=nnl.layers.Dense(node_count=3, activation=nnl.activations.Sigmoid()))
    model.add(layer=nnl.layers.Dense(node_count=5, activation=nnl.activations.Sigmoid()))

    optimizer = nnl.optimizers.RMSprop()
    loss = nnl.losses.MSE()

    model.compile(optimizer=optimizer, loss=loss)

    res = model.predict(x=np.array([1, 1]))
    print(res)


if __name__ == '__main__':
    main()
