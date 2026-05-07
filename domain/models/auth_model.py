from datetime import datetime

class Auth:
    def __init__(
        self,
        id_auth: int,
        id_usuario: int,
        codigo_verificacion: str,
        codigo_expiracion: datetime,
        intentos: int = 0
    ):
        self.id_auth = id_auth
        self.id_usuario = id_usuario
        self.codigo_verificacion = codigo_verificacion
        self.codigo_expiracion = codigo_expiracion
        self.intentos = intentos