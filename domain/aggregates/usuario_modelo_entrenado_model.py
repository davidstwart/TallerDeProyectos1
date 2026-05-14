from datetime import datetime

class UsuarioModeloEntrenado:
    def __init__(
        self,
        id_modelo: int,
        id_usuario: int,
        id_dataset: int,
        nombre_modelo: str,
        ruta_modelo: str,
        precision_modelo: float,
        fecha_entrenamiento: datetime
    ):
        self.id_modelo = id_modelo
        self.id_usuario = id_usuario
        self.id_dataset = id_dataset
        self.nombre_modelo = nombre_modelo
        self.ruta_modelo = ruta_modelo
        self.precision_modelo = precision_modelo
        self.fecha_entrenamiento = fecha_entrenamiento