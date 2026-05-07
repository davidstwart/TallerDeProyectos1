from abc import ABC, abstractmethod


class ITemaInputPort(ABC):

    @abstractmethod
    def create_tema(self, data):
        pass

    @abstractmethod
    def get_temas(self):
        pass