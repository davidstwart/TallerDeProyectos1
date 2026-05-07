from abc import ABC, abstractmethod


class IMachineLearningOutputPort(ABC):

    @abstractmethod
    def save_model(self, model_data):
        pass

    @abstractmethod
    def load_model(self, model_id: int):
        pass