from pydantic import BaseModel

class CreateTemaDTO(BaseModel):

    nombre: str

    descripcion: str