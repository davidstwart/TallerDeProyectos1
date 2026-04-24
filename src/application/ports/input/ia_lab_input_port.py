from abc import ABC, abstractmethod
from typing import Any, Dict

from domain.models.ia_lab_model import DatasetInfo, TrainingResult, PredictionResult


class IIALabInputPort(ABC):
    """Puerto de entrada: define los casos de uso del laboratorio de IA."""

    @abstractmethod
    def generate_simulated_dataset(self, n_samples: int) -> tuple[str, DatasetInfo]:
        """Genera un dataset simulado de rendimiento académico.
        Retorna (session_id, DatasetInfo).
        """

    @abstractmethod
    def upload_csv_dataset(
        self, content: bytes, filename: str, target_column: str
    ) -> tuple[str, list[str], int]:
        """Carga un dataset desde CSV.
        Retorna (session_id, columnas, total_registros).
        """

    @abstractmethod
    def get_dataset_info(self, session_id: str, target_column: str = "aprobado") -> DatasetInfo:
        """Retorna la información exploratoria del dataset de una sesión."""

    @abstractmethod
    def train_model(
        self,
        session_id: str,
        model_name: str,
        params: Dict[str, Any],
        test_size: float,
    ) -> TrainingResult:
        """Entrena un modelo sobre el dataset de la sesión."""

    @abstractmethod
    def predict(
        self,
        session_id: str,
        features: Dict[str, float],
    ) -> PredictionResult:
        """Realiza una predicción usando el modelo entrenado de la sesión."""

    @abstractmethod
    def get_available_models(self) -> list[Dict[str, Any]]:
        """Retorna la lista de modelos disponibles con sus hiperparámetros."""
