import numpy as np
from abc import ABC
from abc import abstractmethod

from nnlibrary.singleton import SingletonMeta


class Losses(metaclass=SingletonMeta):
    def __init__(self):
        self.MSE = MSE()


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def loss(y_target, y_predicted):
        pass

    @staticmethod
    @abstractmethod
    def derivative(y_target, y_predicted):
        pass


class MSE(Loss):
    @staticmethod
    def loss(y_target, y_predicted):
        return np.mean(np.square(y_target - y_predicted))

    @staticmethod
    def derivative(y_target, y_predicted):
        return - 2 * (y_target - y_predicted)