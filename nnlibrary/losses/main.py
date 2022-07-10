import numpy as np

from nnlibrary.losses.abstractions import AbstractLoss


class MSE(AbstractLoss):
    def __call__(self, y_predicted: np.ndarray, y_target: np.ndarray):
        return np.square(y_target - y_predicted) / y_target.shape[-1]