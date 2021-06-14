import numpy as np

import nnlibrary as nnl


def main():
    # nnl = NNLib()

    # nnl.constructor.add(nnl.layers.Input(node_count=3))
    # nnl.constructor.add(nnl.layers.Dense(node_count=5, activation=nnl.activation.HardSigmoid))
    # nnl.constructor.add(nnl.layers.Dense(node_count=6, activation=nnl.activation.Sigmoid))
    # nnl.constructor.add(nnl.layers.Dense(node_count=15, activation=nnl.activation.Sigmoid))
    #
    # loss = nnl.losses.MSE
    # optimizer = nnl.optimizers.Adam(learning_rate=0.1)
    #
    # model = nnl.constructor.compile(loss=loss, optimizer=optimizer)
    model = nnl.nn.Sequential()
    # model.add(nnl.layers.Input(node_count=3))
    # model.add(nnl.layers.Dense(node_count=4, activation=nnl.activation.relu, bias_flag=False))
    # model.add("hello")

    model.add(nnl.layers.Input(node_count=3))
    model.add(nnl.layers.Dense(node_count=5, activation=nnl.activation.sigmoid))
    model.add(nnl.layers.Dense(node_count=15, activation=nnl.activation.sigmoid))
    res = model.show_structure()
    print(res)


if __name__ == '__main__':
    main()
