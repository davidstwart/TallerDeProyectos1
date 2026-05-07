from infrastructure.database.database import (
    SessionLocal
)

from infrastructure.adapters.output.orm.rol_orm import (
    RolORM
)

from domain.models.rol_model import Rol


class MySQLRolRepository:

    def __init__(self):

        self.db = SessionLocal()

    def save(
        self,
        rol: Rol
    ):

        rol_db = RolORM(
            nombre=rol.nombre
        )

        self.db.add(rol_db)

        self.db.commit()

        self.db.refresh(rol_db)

        return Rol(
            id_rol=rol_db.id_rol,
            nombre=rol_db.nombre
        )

    def get_all(self):

        roles_db = (
            self.db.query(RolORM)
            .all()
        )

        return [
            Rol(
                id_rol=r.id_rol,
                nombre=r.nombre
            )
            for r in roles_db
        ]

    def find_by_id(
        self,
        id_rol: int
    ):

        rol_db = (
            self.db.query(RolORM)
            .filter(
                RolORM.id_rol == id_rol
            )
            .first()
        )

        if not rol_db:
            return None

        return Rol(
            id_rol=rol_db.id_rol,
            nombre=rol_db.nombre
        )