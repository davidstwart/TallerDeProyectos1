from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from application.dto.usuario_dto import (
    UsuarioCreateDTO
)

from infrastructure.frameworks.fastapi.dependencies.usuario_dependencies import (
    get_usuario_use_case
)

from infrastructure.security.dependencies import (
    get_current_user
)

router = APIRouter(
    prefix="/usuarios",
    tags=["Usuarios"]
)


@router.post("/")
def create_usuario(
    data: UsuarioCreateDTO,
    current_user=Depends(get_current_user),
    use_case=Depends(get_usuario_use_case)
):

    # ADMIN crea DOCENTES
    if data.id_rol == 2:

        if current_user["rol"] != 1:
            raise HTTPException(
                status_code=403,
                detail="Solo admin crea docentes"
            )

    # DOCENTE crea ESTUDIANTES
    if data.id_rol == 3:

        if current_user["rol"] != 2:
            raise HTTPException(
                status_code=403,
                detail="Solo docente crea estudiantes"
            )

    return use_case.create_usuario(data)


@router.get("/")
def get_usuarios(
    current_user=Depends(get_current_user),
    use_case=Depends(get_usuario_use_case)
):

    return use_case.get_usuarios()

@router.get("/estudiantes")
def get_estudiantes(
    current_user=Depends(get_current_user),
    use_case=Depends(get_usuario_use_case)
):

    return use_case.get_estudiantes()