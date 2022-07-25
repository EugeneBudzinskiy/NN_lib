from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractInitializer(ABC):
    @abstractmethod
    def __call__(self, shape: tuple) -> np.ndarray:
        pass
