from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, y_predicted: np.ndarray, y_target: np.ndarray):
        pass
