from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, y_predicted: ndarray, y_target: ndarray):
        pass
