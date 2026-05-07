from abc import ABC, abstractmethod


class IUsuarioOutputPort(ABC):

    @abstractmethod
    def save(self, usuario):
        pass

    @abstractmethod
    def find_by_email(self, correo: str):
        pass

    @abstractmethod
    def find_by_id(self, usuario_id: int):
        pass

    @abstractmethod
    def get_all(self):
        pass