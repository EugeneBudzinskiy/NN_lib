from abc import ABC
from abc import abstractmethod

from numpy import ndarray


class AbstractActivation(ABC):
    @abstractmethod
    def __call__(self, x: ndarray):
        pass
