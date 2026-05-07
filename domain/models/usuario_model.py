from datetime import date

class Usuario:
    def __init__(
        self,
        id_usuario: int,
        nombres: str,
        apellidos: str,
        fecha_nacimiento: date,
        grado: str,
        seccion: str,
        correo: str,
        celular: str,
        password: str,
        id_rol: int,
        activo: bool = True
    ):
        self.id_usuario = id_usuario
        self.nombres = nombres
        self.apellidos = apellidos
        self.fecha_nacimiento = fecha_nacimiento
        self.grado = grado
        self.seccion = seccion
        self.correo = correo
        self.celular = celular
        self.password = password
        self.id_rol = id_rol
        self.activo = activo