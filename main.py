import numpy as np

from nnlibrary import NNLib


def main():
    nnl = NNLib()

    nnl.constructor.add(nnl.layers.Input(node_count=3))
    nnl.constructor.add(nnl.layers.Dense(node_count=5, activation=nnl.activation.Sigmoid))
    nnl.constructor.add(nnl.layers.Dense(node_count=6, activation=nnl.activation.Sigmoid))
    nnl.constructor.add(nnl.layers.Dense(node_count=15, activation=nnl.activation.Sigmoid))

    loss = nnl.losses.MSE
    optimizer = nnl.optimizers.SGD(learning_rate=0.1)

    model = nnl.constructor.compile(loss=loss, optimizer=optimizer)

    data = np.array([
        [1, -1, 1]
    ])

    target = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    result = model.predict(data)
    # print(result)

    for _ in range(1000):
        model.learn(data, target)


if __name__ == '__main__':
    main()
