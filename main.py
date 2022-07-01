import numpy as np

import nnlibrary as nnl

np.random.seed(13)


def main():
    model = nnl.models.Sequential()

    model.add(nnl.layers.Input(node_count=100))
    model.add(nnl.layers.Dense(node_count=500, activation=nnl.activation_.sigmoid))
    model.add(nnl.layers.Dense(node_count=400, activation=nnl.activation_.sigmoid))
    model.add(nnl.layers.Dense(node_count=300, activation=nnl.activation_.sigmoid))

    optimizer = nnl.optimizers.RMSprop()
    loss = nnl.losses.MSE()
    model.compile(optimizer=optimizer, loss=loss)

    count = 1000
    a = np.random.random((count, 100))
    b = np.random.random((count, 300))

    # res = model.predict(a, verbose=1, steps=1)
    # model.fit(a, b, epochs=5, verbose=0)
    model.fit(a, b, epochs=5)

    # h = model.get_history()
    # for el in h:
    #     print(el, "\n")


if __name__ == '__main__':
    main()
