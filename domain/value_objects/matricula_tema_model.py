from datetime import datetime

class MatriculaTema:
    def __init__(
        self,
        id_matricula: int,
        id_usuario: int,
        id_tema: int,
        fecha_matricula: datetime
    ):
        self.id_matricula = id_matricula
        self.id_usuario = id_usuario
        self.id_tema = id_tema
        self.fecha_matricula = fecha_matricula