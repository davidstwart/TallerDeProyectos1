from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from infrastructure.security.dependencies import (
    get_current_user
)

from infrastructure.frameworks.fastapi.dependencies.dataset_dependencies import (
    get_dataset_use_case
)

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
    use_case=Depends(get_dataset_use_case)
):

    # SOLO DOCENTES
    if current_user["rol"] != 2:

        raise HTTPException(
            status_code=403,
            detail="Solo docentes"
        )

    return use_case.upload_dataset(
        file,
        current_user["id_usuario"]
    )