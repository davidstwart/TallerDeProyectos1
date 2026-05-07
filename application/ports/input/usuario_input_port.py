from abc import ABC, abstractmethod


class IUsuarioInputPort(ABC):

    @abstractmethod
    def create_usuario(self, data):
        pass

    @abstractmethod
    def get_usuarios(self):
        pass

    @abstractmethod
    def get_usuario_by_id(self, usuario_id: int):
        pass