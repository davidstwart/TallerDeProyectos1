from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey
)

from infrastructure.database.base import Base


class AuthORM(Base):
    __tablename__ = "auth"

    id_auth = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    id_usuario = Column(
        Integer,
        ForeignKey("usuario.id_usuario")
    )

    codigo_verificacion = Column(String(10))

    codigo_expiracion = Column(DateTime)

    intentos = Column(Integer, default=0)