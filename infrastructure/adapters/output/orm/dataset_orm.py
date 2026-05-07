from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey
)

from datetime import datetime

from infrastructure.database.base import Base


class DatasetORM(Base):
    __tablename__ = "datasets"

    id_dataset = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    id_usuario = Column(
        Integer,
        ForeignKey("usuario.id_usuario")
    )

    nombre = Column(
        String(255),
        nullable=False
    )

    hash_dataset = Column(
        String(255),
        unique=True,
        nullable=False
    )

    ruta_archivo = Column(
        String(500),
        nullable=False
    )

    fecha_subida = Column(
        DateTime,
        default=datetime.utcnow
    )