from numpy import ndarray
from numpy import square

from nnlibrary.losses.abstractions import AbstractLoss


class MSE(AbstractLoss):
    def __call__(self, y_predicted: ndarray, y_target: ndarray):
        return square(y_target - y_predicted) / y_target.shape[-1]
