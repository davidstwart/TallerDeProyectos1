from fastapi import (
    APIRouter,
    Depends
)

from application.useCases.tema_use_case import (
    TemaUseCase
)

from infrastructure.frameworks.fastapi.dependencies.tema_dependencies import (
    get_tema_use_case
)

from application.dto.tema_dto import (
    CreateTemaDTO
)

router = APIRouter(
    prefix="/temas",
    tags=["Temas"]
)

# =====================================
# CREAR TEMA
# =====================================

@router.post("/")
def create_tema(
    data: CreateTemaDTO,
    use_case: TemaUseCase = Depends(
        get_tema_use_case
    )
):

    return use_case.create_tema(data)

# =====================================
# LISTAR TEMAS
# =====================================

@router.get("/")
def get_temas(
    use_case: TemaUseCase = Depends(
        get_tema_use_case
    )
):

    return use_case.get_temas()