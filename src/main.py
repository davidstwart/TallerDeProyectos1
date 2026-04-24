import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.frameworks.fastapi.ia_lab_router import router as ia_lab_router

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


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "ia-lab"}
