from application.use_cases.tema_use_case import (
    TemaUseCase
)

from infrastructure.adapters.output.repositories.tema_repository import (
    MySQLTemaRepository
)

def get_tema_use_case():

    repository = (
        MySQLTemaRepository()
    )

    return TemaUseCase(
        repository
    )