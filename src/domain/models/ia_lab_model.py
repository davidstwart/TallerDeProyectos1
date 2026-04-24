from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelParams:
    """Parámetros de configuración para un modelo de ML."""
    model_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2


@dataclass
class TrainingResult:
    """Resultado del entrenamiento de un modelo."""
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
    session_id: str


@dataclass
class PredictionResult:
    """Resultado de una predicción."""
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    model_name: str = ""


@dataclass
class DatasetInfo:
    """Información descriptiva de un dataset."""
    total_records: int
    total_features: int
    feature_names: List[str]
    target_classes: Dict[str, int]
    statistics: Dict[str, Any]
    correlation: Dict[str, Any]
    preview: List[Dict[str, Any]]

