import time
from collections import OrderedDict
from typing import Any, Dict, Optional
import pandas as pd

from application.ports.output.ia_lab_output_port import ISessionRepository

_MAX_SESSIONS = 200        # máximo de sesiones simultáneas en memoria
_SESSION_TTL = 3600        # segundos — 1 hora de vida por sesión


class InMemorySessionRepository(ISessionRepository):
    """Adaptador de salida: almacenamiento en memoria de sesiones.

    Limitaciones conocidas:
    - Máximo _MAX_SESSIONS sesiones activas (LRU: se expulsa la más antigua).
    - TTL de _SESSION_TTL segundos por sesión.
    - No persiste entre reinicios del servidor.
    - No es compartido entre múltiples workers de uvicorn (--workers N > 1).
      Para ese caso, reemplazar este adaptador por uno basado en Redis.
    """

    def __init__(self):
        # OrderedDict para LRU: el más reciente al final
        self._dataframes: OrderedDict[str, dict] = OrderedDict()
        self._models: Dict[str, Dict[str, Any]] = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _touch(self, session_id: str) -> None:
        """Mueve la sesión al final (más reciente) y actualiza timestamp."""
        if session_id in self._dataframes:
            self._dataframes.move_to_end(session_id)
            self._dataframes[session_id]["ts"] = time.time()

    def _evict_if_needed(self) -> None:
        """Expulsa sesiones expiradas o la más antigua si se supera el límite."""
        now = time.time()
        # Expirar por TTL
        expired = [
            sid for sid, v in self._dataframes.items()
            if now - v["ts"] > _SESSION_TTL
        ]
        for sid in expired:
            self._dataframes.pop(sid, None)
            self._models.pop(sid, None)

        # Expulsar la más antigua si aún se supera el límite
        while len(self._dataframes) >= _MAX_SESSIONS:
            oldest_sid, _ = self._dataframes.popitem(last=False)
            self._models.pop(oldest_sid, None)

    # ── ISessionRepository ────────────────────────────────────────────────────

    def save_dataframe(self, session_id: str, df: pd.DataFrame, target_column: str) -> None:
        self._evict_if_needed()
        self._dataframes[session_id] = {
            "df": df,
            "target": target_column,
            "ts": time.time(),
        }

    def get_dataframe(self, session_id: str) -> Optional[tuple]:
        entry = self._dataframes.get(session_id)
        if entry is None:
            return None
        self._touch(session_id)
        return entry["df"], entry["target"]

    def save_trained_model(
        self,
        session_id: str,
        model: Any,
        scaler: Any,
        feature_names: list,
        model_name: str,
    ) -> None:
        self._models[session_id] = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": model_name,
        }
        self._touch(session_id)

    def get_trained_model(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._dataframes
