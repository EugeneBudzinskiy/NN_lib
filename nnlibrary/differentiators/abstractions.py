from abc import ABC
from abc import abstractmethod
from typing import Callable

import numpy as np


class AbstractDifferentiator(ABC):
    @abstractmethod
    def __call__(self,
                 func: Callable[[np.ndarray], np.ndarray],
                 x: np.ndarray,
                 epsilon: float = 1e-5) -> np.ndarray:
        pass
