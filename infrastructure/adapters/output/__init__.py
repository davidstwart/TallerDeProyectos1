from infrastructure.database.database import (
    engine
)

from infrastructure.database.base import (
    Base
)

# IMPORTS ORM
from infrastructure.adapters.output.orm.rol_orm import RolORM
from infrastructure.adapters.output.orm.usuario_orm import UsuarioORM
from infrastructure.adapters.output.orm.auth_orm import AuthORM
from infrastructure.adapters.output.orm.tema_orm import TemaORM
from infrastructure.adapters.output.orm.dataset_orm import DatasetORM
from infrastructure.adapters.output.orm.proceso_tema_orm import ProcesoTemaORM
from infrastructure.adapters.output.orm.matricula_tema_orm import MatriculaTemaORM
from infrastructure.adapters.output.orm.usuario_modelo_entrenado_orm import (
    UsuarioModeloEntrenadoORM
)
from infrastructure.adapters.output.orm.usuario_prediccion_orm import (
    UsuarioPrediccionORM
)


def init_db():

    Base.metadata.create_all(
        bind=engine
    )