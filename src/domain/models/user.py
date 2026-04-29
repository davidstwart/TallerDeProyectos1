from dataclasses import dataclass

@dataclass
class User:
    id_usuario: int | None
    email: str
    password: str
    rol: str = "estudiante"