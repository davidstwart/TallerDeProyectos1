from sqlalchemy import (
    Column,
    Integer,
    DateTime,
    ForeignKey
)

from datetime import datetime

from infrastructure.database.base import Base


class MatriculaTemaORM(Base):
    __tablename__ = "matricula_tema"

    id_matricula = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    id_usuario = Column(
        Integer,
        ForeignKey("usuario.id_usuario")
    )

    id_tema = Column(
        Integer,
        ForeignKey("tema.id_tema")
    )

    fecha_matricula = Column(
        DateTime,
        default=datetime.utcnow
    )