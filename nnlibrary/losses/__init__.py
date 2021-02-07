from abc import ABC
from abc import abstractmethod

import numpy as np

from nnlibrary.singleton import SingletonMeta


class Losses(metaclass=SingletonMeta):
    def __init__(self):
        self.MSE = MSE()


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def loss(y_predicted, y_target):
        pass

    @staticmethod
    @abstractmethod
    def derivative(y_predicted, y_target):
        pass


class MSE(Loss):
    @staticmethod
    def loss(y_predicted, y_target):
        return np.mean(np.square(y_target - y_predicted), axis=0)

    @staticmethod
    def derivative(y_predicted, y_target):
        return - 2 * (y_target - y_predicted) / len(y_target)
