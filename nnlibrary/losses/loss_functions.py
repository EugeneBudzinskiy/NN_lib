from abc import ABC
from abc import abstractmethod

from numpy import ndarray
from numpy import square


class Loss(ABC):
    @abstractmethod
    def __call__(self, y_predicted: ndarray, y_target: ndarray):
        pass


class MSE(Loss):
    def __call__(self, y_predicted: ndarray, y_target: ndarray):
        return square(y_target - y_predicted) / y_target.shape[-1]
