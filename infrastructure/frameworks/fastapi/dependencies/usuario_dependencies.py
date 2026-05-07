from infrastructure.adapters.output.repositories.mysql_usuario_repository import (
    MySQLUsuarioRepository
)

from application.useCases.usuario_use_case import (
    UsuarioUseCase
)


def get_usuario_use_case():

    repository = MySQLUsuarioRepository()

    return UsuarioUseCase(repository)