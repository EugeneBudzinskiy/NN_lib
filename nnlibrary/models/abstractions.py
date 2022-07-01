from abc import ABC
from abc import abstractmethod

from numpy import ndarray

from nnlibrary.layers import AbstractLayer
from nnlibrary.optimizers import AbstractOptimizer
from nnlibrary.losses import AbstractLoss


class AbstractModel(ABC):
    @abstractmethod
    def add(self, layer: AbstractLayer):
        pass

    @abstractmethod
    def compile(self,
                optimizer: AbstractOptimizer = None,
                loss: AbstractLoss = None):
        pass

    @abstractmethod
    def predict(self, x: ndarray):
        pass

    @abstractmethod
    def fit(self,
            x: ndarray,
            y: ndarray,
            epoch_number: int = 1,
            batch_size: int = 32,
            shuffle: bool = False):
        pass
