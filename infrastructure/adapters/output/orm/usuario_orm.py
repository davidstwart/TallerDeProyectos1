from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    Boolean,
    ForeignKey
)

from sqlalchemy.orm import relationship

from infrastructure.database.base import Base


class UsuarioORM(Base):
    __tablename__ = "usuario"

    id_usuario = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    nombres = Column(String(100), nullable=False)

    apellidos = Column(String(100), nullable=False)

    fecha_nacimiento = Column(Date)

    grado = Column(String(50))

    seccion = Column(String(20))

    correo = Column(
        String(150),
        unique=True,
        nullable=False
    )

    celular = Column(String(20))

    password = Column(String(255), nullable=False)

    activo = Column(Boolean, default=True)

    id_rol = Column(
        Integer,
        ForeignKey("rol.id_rol")
    )

    rol = relationship("RolORM")