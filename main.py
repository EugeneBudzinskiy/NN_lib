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

    # res = model.predict(x=np.array([1, 1]))
    # print(res)

    # f = lambda x: x[0] ** 3 + 7 * x[1] + 13 * np.sqrt(x[2]) - x[0] * x[1] * x[2]
    f = lambda x: x[0] * x[1] * x[2]
    point = np.array([2, 3, 25])

    old_diff = nnl.differentiators.SimpleDifferentiator()
    diff = nnl.differentiators.Differentiator()

    old_res = old_diff(func=f, x=point)
    res = diff(func=f, x=point)
    print(old_res)
    print(res, np.sum(res))


if __name__ == '__main__':
    main()
