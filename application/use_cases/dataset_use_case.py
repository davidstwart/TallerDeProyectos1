import hashlib
import os

from application.ports.input.dataset_input_port import (
    IDatasetInputPort
)

from domain.models.dataset_model import Dataset


class DatasetUseCase(IDatasetInputPort):

    def __init__(self, dataset_repository):
        self.dataset_repository = dataset_repository

    def upload_dataset(
        self,
        file,
        usuario_id: int
    ):

        content = file.file.read()

        hash_dataset = hashlib.sha256(
            content
        ).hexdigest()

        dataset_existente = (
            self.dataset_repository.find_by_hash(
                hash_dataset
            )
        )

        # SI YA EXISTE
        if dataset_existente:

            return {
                "message": "Dataset ya existente",
                "dataset_id": dataset_existente.id_dataset
            }

        # CREAR DIRECTORIO
        os.makedirs("datasets", exist_ok=True)

        ruta_archivo = f"datasets/{file.filename}"

        with open(ruta_archivo, "wb") as f:
            f.write(content)

        dataset = Dataset(
            id_dataset=None,
            id_usuario=usuario_id,
            nombre=file.filename,
            hash_dataset=hash_dataset,
            ruta_archivo=ruta_archivo,
            fecha_subida=None
        )

        dataset_guardado = (
            self.dataset_repository.save(dataset)
        )

        return {
            "message": "Dataset guardado",
            "dataset_id": dataset_guardado.id_dataset
        }