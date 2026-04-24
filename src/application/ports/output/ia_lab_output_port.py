from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class ISessionRepository(ABC):
    """Puerto de salida: almacenamiento de sesiones (dataset + modelo entrenado)."""

    @abstractmethod
    def save_dataframe(self, session_id: str, df: pd.DataFrame, target_column: str) -> None:
        """Persiste el dataframe asociado a una sesión."""

    @abstractmethod
    def get_dataframe(self, session_id: str) -> Optional[tuple]:
        """Recupera (df, target_column) de la sesión o None si no existe."""

    @abstractmethod
    def save_trained_model(
        self,
        session_id: str,
        model: Any,
        scaler: Any,
        feature_names: list,
        model_name: str,
    ) -> None:
        """Persiste el modelo entrenado y su scaler."""

    @abstractmethod
    def get_trained_model(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Recupera el modelo entrenado de la sesión o None si no existe."""

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Verifica si una sesión existe."""
