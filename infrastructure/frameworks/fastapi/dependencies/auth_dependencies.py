from infrastructure.adapters.output.repositories.mysql_auth_repository import (
    MySQLAuthRepository
)

from infrastructure.adapters.output.repositories.mysql_usuario_repository import (
    MySQLUsuarioRepository
)

from application.useCases.auth_use_case import (
    AuthUseCase
)


def get_auth_use_case():

    auth_repository = (
        MySQLAuthRepository()
    )

    usuario_repository = (
        MySQLUsuarioRepository()
    )

    return AuthUseCase(
        auth_repository,
        usuario_repository
    )