from infrastructure.adapters.output.repositories.mysql_machine_learning_repository import (
    MachineLearningRepository
)

from infrastructure.adapters.output.repositories.session_repository import (
    InMemorySessionRepository
)

from application.use_cases.machine_learning_use_case import (
    MachineLearningUseCase
)


def get_machine_learning_use_case():

    ml_repository = (
        MachineLearningRepository()
    )

    session_repository = (
        InMemorySessionRepository()
    )

    return MachineLearningUseCase(
        ml_repository,
        session_repository
    )