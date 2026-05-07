class Tema:
    def __init__(
        self,
        id_tema: int,
        nombre: str,
        descripcion: str,
        activo: bool = True
    ):
        self.id_tema = id_tema
        self.nombre = nombre
        self.descripcion = descripcion
        self.activo = activo