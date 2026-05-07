from pydantic import BaseModel, EmailStr
from datetime import date
from typing import Optional


class UsuarioCreateDTO(BaseModel):
    nombres: str
    apellidos: str
    fecha_nacimiento: date
    grado: Optional[str] = None
    seccion: Optional[str] = None
    correo: EmailStr
    celular: str
    password: str
    id_rol: int


class UsuarioResponseDTO(BaseModel):
    id_usuario: int
    nombres: str
    apellidos: str
    correo: str
    celular: str
    id_rol: int
    activo: bool

    class Config:
        from_attributes = True