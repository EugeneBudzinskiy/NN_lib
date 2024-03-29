import numpy as np

from nnlibrary import activations
from nnlibrary import reductions
from .abstractions import AbstractLoss


class MeanSquaredError(AbstractLoss):
    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: reductions.AbstractReduction = reductions.ReductionMean()) -> np.ndarray:
        value = np.mean(np.square(y_predicted - y_target), axis=-1).reshape(1, -1)
        return reduction(values=value)


class CategoricalCrossentropy(AbstractLoss):
    def __init__(self,
                 from_logits: bool = False,
                 epsilon: float = 1e-7):
        self.softmax = activations.Softmax()

        self.from_logits = from_logits
        self.epsilon = epsilon

    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: reductions.AbstractReduction = reductions.ReductionMean()) -> np.ndarray:
        if self.from_logits:
            y_predicted = self.softmax(y_predicted)
        else:
            y_predicted /= np.sum(y_predicted, axis=-1).reshape(-1, 1)

        y_predicted = np.maximum(y_predicted, self.epsilon)
        y_predicted = np.minimum(y_predicted, 1. - self.epsilon)

        value = - np.sum(y_target * np.log(y_predicted), axis=-1).reshape(1, -1)
        return reduction(values=value)
