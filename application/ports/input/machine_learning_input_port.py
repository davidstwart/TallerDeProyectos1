from abc import ABC, abstractmethod


class IMachineLearningInputPort(ABC):

    @abstractmethod
    def train_model(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass