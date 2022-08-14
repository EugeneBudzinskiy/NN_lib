from abc import ABC
from abc import abstractmethod

import numpy as np

from nnlibrary.reductions import AbstractReduction
from nnlibrary.reductions import ReductionMean


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self,
                 y_predicted: np.ndarray,
                 y_target: np.ndarray,
                 reduction: AbstractReduction = ReductionMean()) -> np.ndarray:
        pass
