import sys
import os
print("🔥 MAIN EJECUTÁNDOSE")
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# import rutas de la IA
from infrastructure.frameworks.fastapi.ia_lab_router import router as ia_lab_router
# importando ruta de auth
from infrastructure.frameworks.fastapi.auth_router import router as auth_router
# importando ruta /
from infrastructure.frameworks.fastapi.root_router import router as root_router
# importando ruta de health
from infrastructure.frameworks.fastapi.health_router import router as health_router


from pydantic import BaseModel, EmailStr
from infrastructure.adapters.output.mysql_user_repository import MySQLUserRepository
from application.useCases.auth_use_case import AuthUseCase
from enum import Enum
from fastapi import HTTPException, status


app = FastAPI(
    title="Laboratorio Interactivo de IA",
    description=(
        "API REST del Laboratorio Educativo de Inteligencia Artificial. "
        "Permite generar/cargar datasets, entrenar modelos de ML y realizar predicciones."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] es seguro aquí porque no se usan cookies ni credenciales.
# Para producción con autenticación, reemplazar "*" por los orígenes exactos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Manejador global de excepciones no controladas ────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor.",
            "type": type(exc).__name__,
        },
    )

# ── Rutas ─────────────────────────────────────────────────────────────────────
app.include_router(ia_lab_router)
app.include_router(auth_router)
app.include_router(root_router)
app.include_router(health_router)

# =================================================
#roles
class UserRole(str, Enum):
    estudiante = "estudiante"
    profesor = "profesor"

# .Esquemas para recibir datos
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    rol: UserRole = UserRole.estudiante

# .Esquema para Login
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# # ── .Rutas de Autenticación ────────────────────────────────────────────────────


# @app.post("/register", tags=["Auth"], status_code=status.HTTP_201_CREATED)
# def register(user_data: RegisterRequest):
#     try:
#         return auth_service.register(user_data.email, user_data.password, user_data.rol)
#     except Exception as e:
        
#         error_msg = str(e)
#         if "Duplicate entry" in error_msg:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST, 
#                 detail="Este correo electrónico ya está registrado."
#             )
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
#             detail="No se pudo completar el registro. Inténtalo más tarde."
#         )

# @app.post("/login", tags=["Auth"])
# def login(user_data: LoginRequest):
#     result = auth_service.login(user_data.email, user_data.password)
    
#     if not result:
        
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Correo o contraseña incorrectos.",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     return result