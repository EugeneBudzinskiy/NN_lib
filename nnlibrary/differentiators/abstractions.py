from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractDifferentiator(ABC):
    @abstractmethod
    def __call__(self, func: callable, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        pass
