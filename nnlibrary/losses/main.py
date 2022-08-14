import numpy as np

from nnlibrary.differentiators import Gradient
from nnlibrary.losses import AbstractLoss
from nnlibrary.losses import AbstractReduction


class ReductionNone(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values.copy()


class ReductionSum(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.sum(values, axis=-1)


class ReductionMean(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.mean(values, axis=-1)


class MeanSquaredError(AbstractLoss):
    def __init__(self):
        self.gradient = Gradient()

    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: AbstractReduction = ReductionMean()) -> np.ndarray:
        y_predicted = y_predicted if y_predicted.ndim > 1 else y_predicted.reshape(1, -1)
        y_target = y_target if y_target.ndim > 1 else y_target.reshape(1, -1)

        value = np.mean(np.square(y_predicted - y_target), axis=-1).reshape(1, -1)
        return reduction(values=value)
