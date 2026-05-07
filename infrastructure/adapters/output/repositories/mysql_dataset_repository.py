from infrastructure.database.database import (
    SessionLocal
)

from infrastructure.adapters.output.orm.dataset_orm import (
    DatasetORM
)

from domain.models.dataset_model import Dataset

from application.ports.output.dataset_output_port import (
    IDatasetOutputPort
)


class MySQLDatasetRepository(
    IDatasetOutputPort
):

    def __init__(self):

        self.db = SessionLocal()

    def save(
        self,
        dataset: Dataset
    ):

        dataset_db = DatasetORM(
            id_usuario=dataset.id_usuario,
            nombre=dataset.nombre,
            hash_dataset=dataset.hash_dataset,
            ruta_archivo=dataset.ruta_archivo
        )

        self.db.add(dataset_db)

        self.db.commit()

        self.db.refresh(dataset_db)

        return Dataset(
            id_dataset=dataset_db.id_dataset,
            id_usuario=dataset_db.id_usuario,
            nombre=dataset_db.nombre,
            hash_dataset=dataset_db.hash_dataset,
            ruta_archivo=dataset_db.ruta_archivo,
            fecha_subida=dataset_db.fecha_subida
        )

    def find_by_hash(
        self,
        hash_dataset: str
    ):

        dataset_db = (
            self.db.query(DatasetORM)
            .filter(
                DatasetORM.hash_dataset
                == hash_dataset
            )
            .first()
        )

        if not dataset_db:
            return None

        return Dataset(
            id_dataset=dataset_db.id_dataset,
            id_usuario=dataset_db.id_usuario,
            nombre=dataset_db.nombre,
            hash_dataset=dataset_db.hash_dataset,
            ruta_archivo=dataset_db.ruta_archivo,
            fecha_subida=dataset_db.fecha_subida
        )

    def find_by_name(
        self,
        nombre: str
    ):

        dataset_db = (
            self.db.query(DatasetORM)
            .filter(
                DatasetORM.nombre == nombre
            )
            .first()
        )

        if not dataset_db:
            return None

        return Dataset(
            id_dataset=dataset_db.id_dataset,
            id_usuario=dataset_db.id_usuario,
            nombre=dataset_db.nombre,
            hash_dataset=dataset_db.hash_dataset,
            ruta_archivo=dataset_db.ruta_archivo,
            fecha_subida=dataset_db.fecha_subida
        )