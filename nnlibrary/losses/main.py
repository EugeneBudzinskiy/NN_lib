import numpy as np

from nnlibrary.losses import AbstractLoss

# TODO  Loss should normally return single value. And return multiple while `fit`
# TODO  Probably implement reducer (i.e. `mean`, `sum`, `none` ...)


class MeanSquaredError(AbstractLoss):
    def __call__(self, y_predicted: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_predicted = y_predicted if y_predicted.ndim > 1 else y_predicted.reshape(1, -1)
        y_target = y_target if y_target.ndim > 1 else y_target.reshape(1, -1)
        return np.mean(np.square(y_predicted - y_target), axis=1).reshape(1, -1)
