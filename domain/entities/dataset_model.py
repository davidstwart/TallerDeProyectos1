from datetime import datetime

class Dataset:
    def __init__(
        self,
        id_dataset: int,
        id_usuario: int,
        nombre: str,
        hash_dataset: str,
        ruta_archivo: str,
        fecha_subida: datetime
    ):
        self.id_dataset = id_dataset
        self.id_usuario = id_usuario
        self.nombre = nombre
        self.hash_dataset = hash_dataset
        self.ruta_archivo = ruta_archivo
        self.fecha_subida = fecha_subida