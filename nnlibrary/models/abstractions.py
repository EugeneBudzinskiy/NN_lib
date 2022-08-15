from abc import ABC
from abc import abstractmethod

import numpy as np

from nnlibrary.layers import AbstractLayer
from nnlibrary.losses import AbstractLoss
from nnlibrary.optimizers import AbstractOptimizer


class AbstractModel(ABC):
    @abstractmethod
    def add(self, layer: AbstractLayer):
        pass

    @abstractmethod
    def compile(self,
                optimizer: AbstractOptimizer = None,
                loss: AbstractLoss = None):
        pass

    @property
    @abstractmethod
    def is_compiled(self) -> bool:
        pass

    @abstractmethod
    def get_variables(self) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            shuffle: bool = False):
        pass
