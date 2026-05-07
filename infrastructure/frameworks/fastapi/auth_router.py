from fastapi import APIRouter, Depends

from application.dto.auth_dto import (
    LoginDTO,
    RecoverPasswordDTO,
    ResetPasswordDTO
)

from infrastructure.frameworks.fastapi.dependencies.auth_dependencies import (
    get_auth_use_case
)

router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)


@router.post("/login")
def login(
    data: LoginDTO,
    use_case=Depends(get_auth_use_case)
):

    return use_case.login(data)


@router.post("/recover")
def recover_password(
    data: RecoverPasswordDTO,
    use_case=Depends(get_auth_use_case)
):

    return use_case.recover_password(data)


@router.post("/reset")
def reset_password(
    data: ResetPasswordDTO,
    use_case=Depends(get_auth_use_case)
):

    return use_case.reset_password(data)