from sqlalchemy import (
    Column,
    Integer,
    Text,
    DateTime,
    ForeignKey
)

from datetime import datetime

from infrastructure.database.base import Base


class UsuarioPrediccionORM(Base):
    __tablename__ = "usuario_prediccion"

    id_prediccion = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    id_usuario = Column(
        Integer,
        ForeignKey("usuario.id_usuario")
    )

    id_modelo = Column(
        Integer,
        ForeignKey("usuario_modelo_entrenado.id_modelo")
    )

    datos_entrada = Column(Text)

    resultado_prediccion = Column(Text)

    fecha_prediccion = Column(
        DateTime,
        default=datetime.utcnow
    )