from sqlalchemy import Column, Integer, String

from infrastructure.database.base import Base


class RolORM(Base):
    __tablename__ = "rol"

    id_rol = Column(Integer, primary_key=True, autoincrement=True)

    nombre = Column(
        String(50),
        unique=True,
        nullable=False
    )