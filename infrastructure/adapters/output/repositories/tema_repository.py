from infrastructure.database.database import (
    SessionLocal
)

from infrastructure.adapters.output.orm.tema_orm import (
    TemaORM
)

from domain.models.tema_model import (
    Tema
)


class MySQLTemaRepository:

    def __init__(self):

        self.db = SessionLocal()

    def save(
        self,
        tema: Tema
    ):

        tema_db = TemaORM(
            nombre=tema.nombre,
            descripcion=tema.descripcion,
            activo=tema.activo
        )

        self.db.add(tema_db)

        self.db.commit()

        self.db.refresh(tema_db)

        return Tema(
            id_tema=tema_db.id_tema,
            nombre=tema_db.nombre,
            descripcion=tema_db.descripcion,
            activo=tema_db.activo
        )

    def get_all(self):

        temas_db = (
            self.db.query(TemaORM)
            .all()
        )

        return [
            Tema(
                id_tema=t.id_tema,
                nombre=t.nombre,
                descripcion=t.descripcion,
                activo=t.activo
            )
            for t in temas_db
        ]