from datetime import datetime

class UsuarioPrediccion:
    def __init__(
        self,
        id_prediccion: int,
        id_usuario: int,
        id_modelo: int,
        datos_entrada: str,
        resultado_prediccion: str,
        fecha_prediccion: datetime
    ):
        self.id_prediccion = id_prediccion
        self.id_usuario = id_usuario
        self.id_modelo = id_modelo
        self.datos_entrada = datos_entrada
        self.resultado_prediccion = resultado_prediccion
        self.fecha_prediccion = fecha_prediccion