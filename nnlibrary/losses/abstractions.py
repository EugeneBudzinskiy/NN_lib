from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractReduction(ABC):
    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        pass


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: AbstractReduction) -> np.ndarray:
        pass

    @abstractmethod
    def get_gradient(self,
                     y_predicted: np.ndarray,
                     y_target: np.ndarray):
        pass
