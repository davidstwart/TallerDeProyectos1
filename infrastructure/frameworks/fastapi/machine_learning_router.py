from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from application.dto.machine_learning_dto import (
    TrainModelDTO,
    PredictionDTO
)

from infrastructure.security.dependencies import (
    get_current_user
)

from infrastructure.frameworks.fastapi.dependencies.machine_learning_dependencies import (
    get_machine_learning_use_case
)

router = APIRouter(
    prefix="/machine-learning",
    tags=["Machine Learning"]
)


@router.post("/train")
def train_model(
    data: TrainModelDTO,
    current_user=Depends(get_current_user),
    use_case=Depends(
        get_machine_learning_use_case
    )
):

    if current_user["rol"] != 2:

        raise HTTPException(
            status_code=403,
            detail="Solo docentes"
        )

    return use_case.train_model(data)


@router.post("/predict")
def predict(
    data: PredictionDTO,
    current_user=Depends(get_current_user),
    use_case=Depends(
        get_machine_learning_use_case
    )
):

    return use_case.predict(data)