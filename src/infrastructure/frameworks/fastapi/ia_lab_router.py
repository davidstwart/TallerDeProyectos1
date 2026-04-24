from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from application.dto.ia_lab_dto import (
    DatasetInfoResponse,
    GenerateDatasetRequest,
    ModelsInfoResponse,
    PredictRequest,
    PredictResponse,
    TrainModelRequest,
    TrainModelResponse,
    UploadCSVResponse,
)
from application.useCases.ia_lab_use_case import IALabUseCase
from infrastructure.adapters.output.session_repository import InMemorySessionRepository

# ── Dependency injection (singleton in-memory repo) ───────────────────────────
# Un único repositorio compartido por todos los requests del mismo proceso.
# NOTA: con múltiples workers (--workers N) cada proceso tiene su propio repo;
#       usar un store externo (Redis) si se necesita escalar horizontalmente.
_repo = InMemorySessionRepository()


def get_use_case() -> IALabUseCase:
    return IALabUseCase(_repo)


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/v1/ia-lab", tags=["IA Lab"])


# ── 1. Modelos disponibles ────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=ModelsInfoResponse,
    summary="Listar modelos de ML disponibles",
)
def list_models(uc: IALabUseCase = Depends(get_use_case)):
    """Retorna los modelos disponibles con sus hiperparámetros configurables."""
    return ModelsInfoResponse(models=uc.get_available_models())


# ── 2. Dataset: generar simulado ──────────────────────────────────────────────

@router.post(
    "/dataset/generate",
    response_model=DatasetInfoResponse,
    summary="Generar dataset simulado de rendimiento académico",
    status_code=201,
)
def generate_dataset(
    body: GenerateDatasetRequest,
    uc: IALabUseCase = Depends(get_use_case),
):
    """Genera un dataset sintético y retorna su `session_id` junto con
    la información exploratoria. Usa el `session_id` en las siguientes llamadas."""
    session_id, info = uc.generate_simulated_dataset(body.n_samples)
    return DatasetInfoResponse(
        session_id=session_id,
        total_records=info.total_records,
        total_features=info.total_features,
        feature_names=info.feature_names,
        target_classes=info.target_classes,
        statistics=info.statistics,
        correlation=info.correlation,
        preview=info.preview,
    )


# ── 3. Dataset: cargar CSV ────────────────────────────────────────────────────

@router.post(
    "/dataset/upload",
    response_model=UploadCSVResponse,
    summary="Cargar dataset desde archivo CSV",
    status_code=201,
)
async def upload_dataset(
    file: UploadFile = File(..., description="Archivo CSV con los datos"),
    target_column: str = Query(default="aprobado",
                               description="Nombre de la columna objetivo"),
    uc: IALabUseCase = Depends(get_use_case),
):
    """Sube un CSV propio. Retorna `session_id` y listado de columnas detectadas."""
    content = await file.read()
    try:
        session_id, columns, total = uc.upload_csv_dataset(
            content, file.filename or "data.csv", target_column
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar el CSV: {exc}")
    return UploadCSVResponse(
        session_id=session_id,
        message="CSV cargado correctamente.",
        columns=columns,
        total_records=total,
    )


# ── 4. Dataset: explorar ──────────────────────────────────────────────────────

@router.get(
    "/dataset/{session_id}/info",
    response_model=DatasetInfoResponse,
    summary="Explorar dataset de una sesión",
)
def get_dataset_info(
    session_id: str,
    target_column: str = Query(default="aprobado",
                               description="Columna objetivo del dataset"),
    uc: IALabUseCase = Depends(get_use_case),
):
    """Retorna estadísticas descriptivas, correlación y vista previa del dataset."""
    try:
        info = uc.get_dataset_info(session_id, target_column)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return DatasetInfoResponse(
        session_id=session_id,
        total_records=info.total_records,
        total_features=info.total_features,
        feature_names=info.feature_names,
        target_classes=info.target_classes,
        statistics=info.statistics,
        correlation=info.correlation,
        preview=info.preview,
    )


# ── 5. Entrenar modelo ────────────────────────────────────────────────────────

@router.post(
    "/train",
    response_model=TrainModelResponse,
    summary="Entrenar un modelo de ML",
    status_code=201,
)
def train_model(
    body: TrainModelRequest,
    uc: IALabUseCase = Depends(get_use_case),
):
    """Entrena el modelo seleccionado sobre el dataset de la sesión.
    Retorna métricas de evaluación completas y matriz de confusión."""
    try:
        result = uc.train_model(
            session_id=body.session_id,
            model_name=body.model_name,
            params=body.params,
            test_size=body.test_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return TrainModelResponse(
        session_id=result.session_id,
        model_name=result.model_name,
        params=result.params,
        accuracy=result.accuracy,
        precision=result.precision,
        recall=result.recall,
        f1_score=result.f1_score,
        confusion_matrix=result.confusion_matrix,
        classification_report=result.classification_report,
        feature_names=result.feature_names,
        train_samples=result.train_samples,
        test_samples=result.test_samples,
    )


# ── 6. Predecir ───────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Realizar una predicción con el modelo entrenado",
)
def predict(
    body: PredictRequest,
    uc: IALabUseCase = Depends(get_use_case),
):
    """Recibe los valores de las features y retorna la predicción con probabilidades."""
    try:
        result = uc.predict(session_id=body.session_id, features=body.features)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        session_id=body.session_id,
        prediction=result.prediction,
        probabilities=result.probabilities,
        model_name=result.model_name,
    )
