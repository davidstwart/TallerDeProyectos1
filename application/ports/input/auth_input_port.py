from abc import ABC, abstractmethod


class IAuthInputPort(ABC):

    @abstractmethod
    def login(self, data):
        pass

    @abstractmethod
    def recover_password(self, data):
        pass

    @abstractmethod
    def reset_password(self, data):
        pass