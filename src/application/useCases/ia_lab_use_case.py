import io
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from application.ports.input.ia_lab_input_port import IIALabInputPort
from application.ports.output.ia_lab_output_port import ISessionRepository
from domain.models.ia_lab_model import (
    DatasetInfo,
    PredictionResult,
    TrainingResult,
)


class IALabUseCase(IIALabInputPort):
    """Implementación de los casos de uso del Laboratorio de IA."""

    AVAILABLE_MODELS = [
        {
            "name": "Logistic Regression",
            "description": "Traza una línea que separa las clases. Ideal para datos linealmente separables.",
            "hyperparameters": {
                "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0,
                      "description": "Regularización inversa"},
                "max_iter": {"type": "int", "min": 100, "max": 2000, "default": 1000,
                             "description": "Máximo de iteraciones"},
                "solver": {"type": "enum",
                           "options": ["lbfgs", "liblinear", "newton-cg", "saga"],
                           "default": "lbfgs"},
            },
        },
        {
            "name": "Decision Tree",
            "description": "Crea reglas tipo 'si X > 5 entonces...'. Útil con reglas claras.",
            "hyperparameters": {
                "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
                "criterion": {"type": "enum", "options": ["gini", "entropy"], "default": "gini"},
            },
        },
        {
            "name": "Random Forest",
            "description": "Combina muchos árboles de decisión. Reduce overfitting.",
            "hyperparameters": {
                "n_estimators": {"type": "int", "min": 10, "max": 300, "default": 100},
                "max_depth": {"type": "int", "min": 1, "max": 20, "default": 5},
                "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2},
            },
        },
        {
            "name": "SVM",
            "description": "Busca el margen máximo entre clases. Ideal para datasets pequeños/medianos.",
            "hyperparameters": {
                "C": {"type": "float", "min": 0.01, "max": 10.0, "default": 1.0},
                "kernel": {"type": "enum",
                           "options": ["rbf", "linear", "poly", "sigmoid"],
                           "default": "rbf"},
            },
        },
        {
            "name": "KNN",
            "description": "Clasifica según los K vecinos más cercanos.",
            "hyperparameters": {
                "n_neighbors": {"type": "int", "min": 1, "max": 21, "default": 5},
                "weights": {"type": "enum", "options": ["uniform", "distance"],
                            "default": "uniform"},
                "metric": {"type": "enum",
                           "options": ["euclidean", "manhattan", "minkowski"],
                           "default": "euclidean"},
            },
        },
    ]

    def __init__(self, session_repo: ISessionRepository):
        self._repo = session_repo

    # ── Dataset ───────────────────────────────────────────────────────────────

    def generate_simulated_dataset(self, n_samples: int) -> tuple:
        df = self._build_simulated_df(n_samples)
        session_id = str(uuid.uuid4())
        self._repo.save_dataframe(session_id, df, "aprobado")
        info = self._build_dataset_info(df, "aprobado")
        return session_id, info

    def upload_csv_dataset(
        self, content: bytes, filename: str, target_column: str = "aprobado"
    ) -> tuple:
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError("El archivo CSV está vacío o no contiene filas de datos.")
        if target_column and target_column not in df.columns:
            raise ValueError(
                f"La columna objetivo '{target_column}' no existe en el CSV. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )
        session_id = str(uuid.uuid4())
        self._repo.save_dataframe(session_id, df, target_column)
        return session_id, df.columns.tolist(), len(df)

    def get_dataset_info(self, session_id: str, target_column: str = "aprobado") -> DatasetInfo:
        result = self._repo.get_dataframe(session_id)
        if result is None:
            raise ValueError(f"Sesión '{session_id}' no encontrada.")
        df, _ = result
        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no existe en el dataset.")
        return self._build_dataset_info(df, target_column)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_model(
        self,
        session_id: str,
        model_name: str,
        params: Dict[str, Any],
        test_size: float,
    ) -> TrainingResult:
        result = self._repo.get_dataframe(session_id)
        if result is None:
            raise ValueError(f"Sesión '{session_id}' no encontrada.")
        df, target_column = result

        if not target_column:
            target_column = "aprobado"

        X = df.drop(target_column, axis=1).select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            raise ValueError("No hay columnas numéricas disponibles como features.")
        y = df[target_column]

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if y.nunique() < 2:
            raise ValueError(
                "La variable objetivo debe tener al menos 2 clases distintas en el dataset."
            )

        # stratify requiere al menos 2 muestras por clase; si no, entrenar sin stratify
        min_class_count = y.value_counts().min()
        use_stratify = min_class_count >= 2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if use_stratify else None,
            )
        except ValueError as exc:
            raise ValueError(
                f"No se pudo dividir el dataset: {exc}. "
                "Verifica que haya suficientes muestras por clase o reduce el porcentaje de prueba."
            )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = self._build_model(model_name, params)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        classes = sorted(y_test.unique().tolist())
        is_binary = len(classes) == 2
        avg = "binary" if is_binary else "weighted"
        pos_label = classes[1] if is_binary else None

        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        report = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in classes],
            output_dict=True,
            zero_division=0,
        )

        # Persist trained model
        self._repo.save_trained_model(
            session_id, model, scaler, X.columns.tolist(), model_name
        )

        return TrainingResult(
            model_name=model_name,
            params=params,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, average=avg,
                                            pos_label=pos_label, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, average=avg,
                                      pos_label=pos_label, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred, average=avg,
                                    pos_label=pos_label, zero_division=0)),
            confusion_matrix=cm,
            classification_report=report,
            feature_names=X.columns.tolist(),
            train_samples=len(X_train),
            test_samples=len(X_test),
            session_id=session_id,
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, session_id: str, features: Dict[str, float]) -> PredictionResult:
        model_data = self._repo.get_trained_model(session_id)
        if model_data is None:
            raise ValueError(f"No hay modelo entrenado para la sesión '{session_id}'.")

        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_names = model_data["feature_names"]
        model_name = model_data["model_name"]

        row = pd.DataFrame([{f: features.get(f, 0.0) for f in feature_names}])
        row_scaled = scaler.transform(row)
        prediction = model.predict(row_scaled)[0]

        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_scaled)[0]
            probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}

        return PredictionResult(
            prediction=prediction.item() if hasattr(prediction, "item") else prediction,
            probabilities=probabilities,
            model_name=model_name,
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_available_models(self) -> List[Dict[str, Any]]:
        return self.AVAILABLE_MODELS

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_simulated_df(n: int) -> pd.DataFrame:
        # Usar default_rng en lugar de np.random.seed — thread-safe y reproducible
        rng = np.random.default_rng(42)
        data = {
            "horas_estudio": np.round(rng.uniform(0, 30, n), 1),
            "asistencia": np.round(rng.uniform(40, 100, n), 1),
            "promedio_previo": np.round(rng.uniform(5, 20, n), 1),
            "horas_sueno": np.round(rng.uniform(3, 10, n), 1),
            "actividades_extra": rng.choice([0, 1], n, p=[0.4, 0.6]),
            "nivel_socioeconomico": rng.choice([1, 2, 3], n, p=[0.3, 0.5, 0.2]),
            "acceso_internet": rng.choice([0, 1], n, p=[0.2, 0.8]),
        }
        df = pd.DataFrame(data)
        score = (
            0.25 * df["horas_estudio"] / 30
            + 0.20 * df["asistencia"] / 100
            + 0.25 * df["promedio_previo"] / 20
            + 0.10 * df["horas_sueno"] / 10
            + 0.08 * df["actividades_extra"]
            + 0.07 * df["nivel_socioeconomico"] / 3
            + 0.05 * df["acceso_internet"]
        )
        noise = rng.normal(0, 0.05, n)
        df["aprobado"] = (score + noise >= 0.55).astype(int)
        return df

    @staticmethod
    def _build_dataset_info(df: pd.DataFrame, target_column: str) -> DatasetInfo:
        numeric_df = df.select_dtypes(include=[np.number])
        stats = numeric_df.describe().round(4).to_dict()
        corr = numeric_df.corr().round(4).to_dict() if numeric_df.shape[1] >= 2 else {}
        target_counts = df[target_column].value_counts().to_dict()
        preview = df.head(20).fillna("").to_dict(orient="records")
        features = [c for c in df.columns if c != target_column]
        return DatasetInfo(
            total_records=len(df),
            total_features=len(features),
            feature_names=features,
            target_classes={str(k): int(v) for k, v in target_counts.items()},
            statistics={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in stats.items()},
            correlation={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in corr.items()},
            preview=[{str(k): v for k, v in row.items()} for row in preview],
        )

    @staticmethod
    def _build_model(name: str, params: Dict[str, Any]):
        if name == "Logistic Regression":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                solver=params.get("solver", "lbfgs"),
                random_state=42,
            )
        if name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=params.get("max_depth", 5),
                min_samples_split=params.get("min_samples_split", 2),
                criterion=params.get("criterion", "gini"),
                random_state=42,
            )
        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=42,
            )
        if name == "SVM":
            return SVC(
                C=params.get("C", 1.0),
                kernel=params.get("kernel", "rbf"),
                probability=True,
                random_state=42,
            )
        if name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
                metric=params.get("metric", "euclidean"),
            )
        raise ValueError(f"Modelo '{name}' no reconocido.")
