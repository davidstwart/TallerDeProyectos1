from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from enum import Enum

from infrastructure.adapters.output.mysql_user_repository import MySQLUserRepository
from application.useCases.auth_use_case import AuthUseCase

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


# ── Roles ─────────────────────────────────────────────
class UserRole(str, Enum):
    estudiante = "estudiante"
    profesor = "profesor"


# ── DTOs ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    rol: UserRole = UserRole.estudiante


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── Dependency ────────────────────────────────────────
def get_auth_service():
    try:
        repo = MySQLUserRepository()
        return AuthUseCase(repo)
    except Exception as e:
        print("MYSQL ERROR:", e)
        raise HTTPException(status_code=500, detail="DB no disponible")


# ── ENDPOINTS ─────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    user_data: RegisterRequest,
    service: AuthUseCase = Depends(get_auth_service),
):
    try:
        return service.register(
            user_data.email,
            user_data.password,
            user_data.rol,
        )
    except Exception as e:
        if "Duplicate entry" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Correo ya registrado",
            )
        raise HTTPException(status_code=500, detail="Error en registro")


@router.post("/login")
def login(
    user_data: LoginRequest,
    service: AuthUseCase = Depends(get_auth_service),
):
    result = service.login(user_data.email, user_data.password)

    if not result:
        raise HTTPException(
            status_code=401,
            detail="Credenciales incorrectas",
        )

    return result


@router.get("/test")
def test():
    return {"msg": "auth funcionando"}