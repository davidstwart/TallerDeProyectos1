from sqlalchemy.orm import Session

from infrastructure.database.database import (
    SessionLocal
)

from infrastructure.adapters.output.orm.usuario_orm import (
    UsuarioORM
)

from domain.models.usuario_model import Usuario

from application.ports.output.usuario_output_port import (
    IUsuarioOutputPort
)


class MySQLUsuarioRepository(
    IUsuarioOutputPort
):

    def __init__(self):

        self.db: Session = SessionLocal()

    def save(
        self,
        usuario: Usuario
    ):

        usuario_db = UsuarioORM(
            nombres=usuario.nombres,
            apellidos=usuario.apellidos,
            fecha_nacimiento=usuario.fecha_nacimiento,
            grado=usuario.grado,
            seccion=usuario.seccion,
            correo=usuario.correo,
            celular=usuario.celular,
            password=usuario.password,
            id_rol=usuario.id_rol,
            activo=usuario.activo
        )

        self.db.add(usuario_db)

        self.db.commit()

        self.db.refresh(usuario_db)

        return Usuario(
            id_usuario=usuario_db.id_usuario,
            nombres=usuario_db.nombres,
            apellidos=usuario_db.apellidos,
            fecha_nacimiento=usuario_db.fecha_nacimiento,
            grado=usuario_db.grado,
            seccion=usuario_db.seccion,
            correo=usuario_db.correo,
            celular=usuario_db.celular,
            password=usuario_db.password,
            id_rol=usuario_db.id_rol,
            activo=usuario_db.activo
        )

    def find_by_email(
        self,
        correo: str
    ):

        usuario_db = (
            self.db.query(UsuarioORM)
            .filter(
                UsuarioORM.correo == correo
            )
            .first()
        )

        if not usuario_db:
            return None

        return Usuario(
            id_usuario=usuario_db.id_usuario,
            nombres=usuario_db.nombres,
            apellidos=usuario_db.apellidos,
            fecha_nacimiento=usuario_db.fecha_nacimiento,
            grado=usuario_db.grado,
            seccion=usuario_db.seccion,
            correo=usuario_db.correo,
            celular=usuario_db.celular,
            password=usuario_db.password,
            id_rol=usuario_db.id_rol,
            activo=usuario_db.activo
        )

    def find_by_id(
        self,
        usuario_id: int
    ):

        usuario_db = (
            self.db.query(UsuarioORM)
            .filter(
                UsuarioORM.id_usuario == usuario_id
            )
            .first()
        )

        if not usuario_db:
            return None

        return Usuario(
            id_usuario=usuario_db.id_usuario,
            nombres=usuario_db.nombres,
            apellidos=usuario_db.apellidos,
            fecha_nacimiento=usuario_db.fecha_nacimiento,
            grado=usuario_db.grado,
            seccion=usuario_db.seccion,
            correo=usuario_db.correo,
            celular=usuario_db.celular,
            password=usuario_db.password,
            id_rol=usuario_db.id_rol,
            activo=usuario_db.activo
        )

    def get_all(self):

        usuarios_db = (
            self.db.query(UsuarioORM)
            .all()
        )

        return [
            Usuario(
                id_usuario=u.id_usuario,
                nombres=u.nombres,
                apellidos=u.apellidos,
                fecha_nacimiento=u.fecha_nacimiento,
                grado=u.grado,
                seccion=u.seccion,
                correo=u.correo,
                celular=u.celular,
                password=u.password,
                id_rol=u.id_rol,
                activo=u.activo
            )
            for u in usuarios_db
        ]
    
    def get_estudiantes(self):

        usuarios_db = (
            self.db.query(UsuarioORM)
            .filter(
                UsuarioORM.id_rol == 3
            )
            .all()
        )

        return [
            Usuario(
                id_usuario=u.id_usuario,
                nombres=u.nombres,
                apellidos=u.apellidos,
                fecha_nacimiento=u.fecha_nacimiento,
                grado=u.grado,
                seccion=u.seccion,
                correo=u.correo,
                celular=u.celular,
                password=u.password,
                id_rol=u.id_rol,
                activo=u.activo
            )
            for u in usuarios_db
        ]