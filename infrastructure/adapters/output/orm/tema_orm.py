from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean
)

from infrastructure.database.base import Base


class TemaORM(Base):
    __tablename__ = "tema"

    id_tema = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    nombre = Column(
        String(100),
        unique=True,
        nullable=False
    )

    descripcion = Column(String(500))

    activo = Column(Boolean, default=True)