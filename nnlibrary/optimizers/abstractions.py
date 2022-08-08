from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractOptimizer(ABC):
    @abstractmethod
    def __call__(self, gradient_vector: np.ndarray) -> np.ndarray:
        pass
