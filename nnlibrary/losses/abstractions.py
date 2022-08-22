from abc import ABC
from abc import abstractmethod

import numpy as np

from nnlibrary import reductions


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: reductions.AbstractReduction = reductions.ReductionMean()) -> np.ndarray:
        pass
