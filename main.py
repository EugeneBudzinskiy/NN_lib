import numpy as np
import nnlibrary as nnl

np.random.seed(13)


def main():
    model = nnl.nn.Sequential()

    model.add(nnl.layers.Input(node_count=100))
    model.add(nnl.layers.Dense(node_count=80, activation=nnl.activation.sigmoid))
    model.add(nnl.layers.Dense(node_count=60, activation=nnl.activation.sigmoid))
    model.add(nnl.layers.Dense(node_count=30, activation=nnl.activation.sigmoid))

    model.compile(optimizer='RMSprop', loss='MSE')

    a = np.random.random((5000, 100))
    b = np.random.random((5000, 30))

    res = model.predict(a)
    model.fit(a, b, batch_size=2, epochs=5)


if __name__ == '__main__':
    main()
