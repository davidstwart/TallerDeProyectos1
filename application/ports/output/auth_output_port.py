from abc import ABC, abstractmethod


class IAuthOutputPort(ABC):

    @abstractmethod
    def save_code(self, auth):
        pass

    @abstractmethod
    def verify_code(self, correo: str, codigo: str):
        pass