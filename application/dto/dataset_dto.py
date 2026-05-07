from pydantic import BaseModel


class DatasetResponseDTO(BaseModel):
    id_dataset: int
    nombre: str
    ruta_archivo: str

    class Config:
        from_attributes = True