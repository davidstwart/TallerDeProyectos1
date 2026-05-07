import sys
import os

print("🔥 MAIN EJECUTÁNDOSE")

sys.path.insert(
    0,
    os.path.dirname(__file__)
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# DB
from infrastructure.adapters.output.init_db import (
    init_db
)

# ROUTERS
from infrastructure.frameworks.fastapi.auth_router import (
    router as auth_router
)

from infrastructure.frameworks.fastapi.usuario_router import (
    router as usuario_router
)

from infrastructure.frameworks.fastapi.rol_router import (
    router as rol_router
)

from infrastructure.frameworks.fastapi.tema_router import (
    router as tema_router
)

from infrastructure.frameworks.fastapi.dataset_router import (
    router as dataset_router
)

from infrastructure.frameworks.fastapi.machine_learning_router import (
    router as ml_router
)

from infrastructure.frameworks.fastapi.ia_lab_router import (
    router as ia_lab_router
)

from infrastructure.frameworks.fastapi.root_router import (
    router as root_router
)

from infrastructure.frameworks.fastapi.health_router import (
    router as health_router
)

# ==================================================
# INIT DB
# ==================================================

init_db()

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Laboratorio Interactivo de IA",
    description=(
        "API REST del Laboratorio Educativo "
        "de Inteligencia Artificial."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# GLOBAL EXCEPTION HANDLER
# ==================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception
):

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "type": type(exc).__name__
        }
    )

# ==================================================
# ROUTERS
# ==================================================

app.include_router(root_router)

app.include_router(health_router)

app.include_router(auth_router)

app.include_router(usuario_router)

app.include_router(rol_router)

app.include_router(tema_router)

app.include_router(dataset_router)

app.include_router(ml_router)

app.include_router(ia_lab_router)