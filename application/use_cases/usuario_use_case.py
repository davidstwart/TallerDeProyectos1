from application.ports.input.usuario_input_port import (
    IUsuarioInputPort
)
from infrastructure.security.password_manager import (
    hash_password
)

from domain.models.usuario_model import Usuario


class UsuarioUseCase(IUsuarioInputPort):

    def __init__(self, usuario_repository):
        self.usuario_repository = usuario_repository

    def create_usuario(self, data):

        hashed_password = hash_password(
            data.password
        )

        usuario = Usuario(
            id_usuario=None,
            nombres=data.nombres,
            apellidos=data.apellidos,
            fecha_nacimiento=data.fecha_nacimiento,
            grado=data.grado,
            seccion=data.seccion,
            correo=data.correo,
            celular=data.celular,
            password=hashed_password,
            id_rol=data.id_rol
        )

        return self.usuario_repository.save(usuario)

    def get_usuarios(self):
        return self.usuario_repository.get_all()

    def get_usuario_by_id(self, usuario_id: int):
        return self.usuario_repository.find_by_id(
            usuario_id
        )
    
    def get_estudiantes(self):

        return (
            self.usuario_repository
            .get_estudiantes()
        )