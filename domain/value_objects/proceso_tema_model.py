class ProcesoTema:
    def __init__(
        self,
        id_proceso: int,
        id_usuario: int,
        id_tema: int,
        porcentaje_avance: float,
        completado: bool = False
    ):
        self.id_proceso = id_proceso
        self.id_usuario = id_usuario
        self.id_tema = id_tema
        self.porcentaje_avance = porcentaje_avance
        self.completado = completado