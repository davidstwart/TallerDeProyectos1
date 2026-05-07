import os
import time
import joblib

from typing import Any, Dict, Optional
from collections import OrderedDict

import pandas as pd

from application.ports.output.session_output_port import (
    ISessionRepository
)

# CONFIGURACIÓN
_MAX_SESSIONS = 200
_SESSION_TTL = 3600  # 1 hora

MODEL_DIR = "models"


class InMemorySessionRepository(
    ISessionRepository
):

    def __init__(self):

        # DATASETS EN MEMORIA
        self._dataframes: OrderedDict[
            str,
            dict
        ] = OrderedDict()

        # MODELOS EN MEMORIA
        self._models: Dict[
            str,
            Dict[str, Any]
        ] = {}

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

    # ==================================================
    # HELPERS
    # ==================================================

    def _touch(
        self,
        session_id: str
    ):

        if session_id in self._dataframes:

            self._dataframes.move_to_end(
                session_id
            )

            self._dataframes[session_id][
                "ts"
            ] = time.time()

    def _evict_if_needed(self):

        now = time.time()

        # ELIMINAR EXPIRADOS
        expired = [
            sid
            for sid, value in self._dataframes.items()
            if now - value["ts"] > _SESSION_TTL
        ]

        for sid in expired:

            self._dataframes.pop(
                sid,
                None
            )

            self._models.pop(
                sid,
                None
            )

        # LRU
        while len(self._dataframes) >= _MAX_SESSIONS:

            oldest_sid, _ = (
                self._dataframes.popitem(
                    last=False
                )
            )

            self._models.pop(
                oldest_sid,
                None
            )

    # ==================================================
    # DATAFRAME
    # ==================================================

    def save_dataframe(
        self,
        session_id: str,
        df: pd.DataFrame,
        target_column: str
    ):

        self._evict_if_needed()

        self._dataframes[session_id] = {
            "df": df,
            "target": target_column,
            "ts": time.time()
        }

    def get_dataframe(
        self,
        session_id: str
    ) -> Optional[tuple]:

        entry = self._dataframes.get(
            session_id
        )

        if entry is None:
            return None

        self._touch(session_id)

        return (
            entry["df"],
            entry["target"]
        )

    # ==================================================
    # MODELOS EN MEMORIA
    # ==================================================

    def save_trained_model(
        self,
        session_id: str,
        model: Any,
        scaler: Any,
        feature_names: list,
        model_name: str
    ):

        self._models[session_id] = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": model_name
        }

        self._touch(session_id)

    def get_trained_model(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:

        return self._models.get(
            session_id
        )

    # ==================================================
    # PERSISTENCIA MODELOS
    # ==================================================

    def persist_model(
        self,
        model_id: str,
        model: Any,
        scaler: Any,
        feature_names: list,
        model_name: str
    ):

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_id}.pkl"
        )

        payload = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": model_name
        }

        joblib.dump(
            payload,
            model_path
        )

    def load_model(
        self,
        model_id: str
    ) -> Optional[Dict[str, Any]]:

        model_path = os.path.join(
            MODEL_DIR,
            f"{model_id}.pkl"
        )

        if not os.path.exists(model_path):
            return None

        return joblib.load(model_path)

    # ==================================================
    # UTILIDADES
    # ==================================================

    def session_exists(
        self,
        session_id: str
    ) -> bool:

        return (
            session_id in self._dataframes
        )

    def list_models(self):

        return [
            f.replace(".pkl", "")
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl")
        ]