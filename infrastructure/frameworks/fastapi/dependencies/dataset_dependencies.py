from infrastructure.adapters.output.repositories.mysql_dataset_repository import (
    MySQLDatasetRepository
)

from application.use_cases.dataset_use_case import (
    DatasetUseCase
)


def get_dataset_use_case():

    repository = MySQLDatasetRepository()

    return DatasetUseCase(repository)