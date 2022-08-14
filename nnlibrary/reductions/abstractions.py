from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractReduction(ABC):
    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        pass
