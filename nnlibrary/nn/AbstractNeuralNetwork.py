from abc import ABC, abstractmethod


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def predict(self, batch_data):
        pass

    @abstractmethod
    def learn(self, batch_data, batch_target):
        pass