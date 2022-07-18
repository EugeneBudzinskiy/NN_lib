import numpy as np

from nnlibrary.losses import AbstractLoss


class MeanSquaredError(AbstractLoss):
    def __call__(self, y_predicted: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return np.mean(np.square(y_predicted - y_target), axis=1).reshape(1, -1)
