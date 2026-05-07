from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey
)

from datetime import datetime

from infrastructure.database.base import Base


class UsuarioModeloEntrenadoORM(Base):
    __tablename__ = "usuario_modelo_entrenado"

    id_modelo = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    id_usuario = Column(
        Integer,
        ForeignKey("usuario.id_usuario")
    )

    id_dataset = Column(
        Integer,
        ForeignKey("datasets.id_dataset")
    )

    nombre_modelo = Column(String(255))

    ruta_modelo = Column(String(500))

    precision_modelo = Column(Float)

    fecha_entrenamiento = Column(
        DateTime,
        default=datetime.utcnow
    )