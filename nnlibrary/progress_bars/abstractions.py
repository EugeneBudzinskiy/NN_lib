from abc import ABC
from abc import abstractmethod


class AbstractProgressBar(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
