from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractActivation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
