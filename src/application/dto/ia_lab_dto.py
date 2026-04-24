from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Valores permitidos exactos para model_name
ModelName = Literal[
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN",
]

# ── Request DTOs ──────────────────────────────────────────────────────────────

class GenerateDatasetRequest(BaseModel):
    n_samples: int = Field(default=500, ge=100, le=2000,
                           description="Número de estudiantes a generar")


class TrainModelRequest(BaseModel):
    model_name: ModelName = Field(
        description="Nombre del algoritmo. Valores permitidos: "
                    "'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'"
    )
    params: Dict[str, Any] = Field(default_factory=dict,
                                   description="Hiperparámetros del modelo")
    test_size: float = Field(default=0.2, ge=0.1, le=0.4,
                             description="Fracción del dataset para prueba (0.1-0.4)")
    session_id: str = Field(description="ID de sesión que referencia el dataset cargado")


class PredictRequest(BaseModel):
    session_id: str = Field(description="ID de sesión con modelo entrenado")
    features: Dict[str, float] = Field(
        description="Valores de las features para predecir"
    )


# ── Response DTOs ─────────────────────────────────────────────────────────────

class DatasetInfoResponse(BaseModel):
    session_id: str
    total_records: int
    total_features: int
    feature_names: List[str]
    target_classes: Dict[str, int]
    statistics: Dict[str, Any]
    correlation: Dict[str, Any]
    preview: List[Dict[str, Any]]


class TrainModelResponse(BaseModel):
    session_id: str
    model_name: str
    params: Dict[str, Any]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    feature_names: List[str]
    train_samples: int
    test_samples: int


class PredictResponse(BaseModel):
    session_id: str
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    model_name: str


class UploadCSVResponse(BaseModel):
    session_id: str
    message: str
    columns: List[str]
    total_records: int


class ModelsInfoResponse(BaseModel):
    models: List[Dict[str, Any]]
