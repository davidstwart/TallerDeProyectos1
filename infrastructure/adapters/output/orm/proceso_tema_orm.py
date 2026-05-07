from sqlalchemy import (
    Column,
    Integer,
    Float,
    Boolean,
    ForeignKey
)

from infrastructure.database.base import Base


class ProcesoTemaORM(Base):
    __tablename__ = "proceso_tema"

    id_proceso = Column(
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

    porcentaje_avance = Column(Float, default=0)

    completado = Column(Boolean, default=False)