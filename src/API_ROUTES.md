# API Reference — Laboratorio Interactivo de IA

**Base URL:** `http://localhost:8000`  
**Prefijo de todas las rutas:** `/api/v1/ia-lab`  
**Documentación interactiva:** `GET /docs` (Swagger UI) · `GET /redoc`

---

## Flujo de uso recomendado

```
1. GET  /api/v1/ia-lab/models          → conocer modelos disponibles
2. POST /api/v1/ia-lab/dataset/generate  ← o → POST /api/v1/ia-lab/dataset/upload
3. GET  /api/v1/ia-lab/dataset/{session_id}/info
4. POST /api/v1/ia-lab/train
5. POST /api/v1/ia-lab/predict
```

---

## Endpoints

### 1. Health Check

| Campo  | Valor |
|--------|-------|
| Método | `GET` |
| Ruta   | `/health` |

**Response `200`**
```json
{ "status": "ok", "service": "ia-lab" }
```

---

### 2. Listar modelos disponibles

| Campo  | Valor |
|--------|-------|
| Método | `GET` |
| Ruta   | `/api/v1/ia-lab/models` |

**Response `200`**
```json
{
  "models": [
    {
      "name": "Logistic Regression",
      "description": "...",
      "hyperparameters": {
        "C":        { "type": "float", "min": 0.01, "max": 10.0, "default": 1.0 },
        "max_iter": { "type": "int",   "min": 100,  "max": 2000, "default": 1000 },
        "solver":   { "type": "enum",  "options": ["lbfgs","liblinear","newton-cg","saga"], "default": "lbfgs" }
      }
    },
    {
      "name": "Decision Tree",
      "hyperparameters": {
        "max_depth":         { "type": "int",  "min": 1, "max": 20, "default": 5 },
        "min_samples_split": { "type": "int",  "min": 2, "max": 20, "default": 2 },
        "criterion":         { "type": "enum", "options": ["gini","entropy"], "default": "gini" }
      }
    },
    {
      "name": "Random Forest",
      "hyperparameters": {
        "n_estimators":      { "type": "int", "min": 10, "max": 300, "default": 100 },
        "max_depth":         { "type": "int", "min": 1,  "max": 20,  "default": 5 },
        "min_samples_split": { "type": "int", "min": 2,  "max": 20,  "default": 2 }
      }
    },
    {
      "name": "SVM",
      "hyperparameters": {
        "C":      { "type": "float", "min": 0.01, "max": 10.0, "default": 1.0 },
        "kernel": { "type": "enum",  "options": ["rbf","linear","poly","sigmoid"], "default": "rbf" }
      }
    },
    {
      "name": "KNN",
      "hyperparameters": {
        "n_neighbors": { "type": "int",  "min": 1,  "max": 21, "default": 5 },
        "weights":     { "type": "enum", "options": ["uniform","distance"], "default": "uniform" },
        "metric":      { "type": "enum", "options": ["euclidean","manhattan","minkowski"], "default": "euclidean" }
      }
    }
  ]
}
```

---

### 3. Generar dataset simulado

| Campo       | Valor |
|-------------|-------|
| Método      | `POST` |
| Ruta        | `/api/v1/ia-lab/dataset/generate` |
| Status code | `201` |

**Request body** (`application/json`)
```json
{ "n_samples": 500 }
```

| Campo      | Tipo  | Rango      | Default | Descripción |
|------------|-------|------------|---------|-------------|
| `n_samples`| int   | 100 – 2000 | 500     | Número de estudiantes a generar |

**Response `201`**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "total_records": 500,
  "total_features": 7,
  "feature_names": ["horas_estudio","asistencia","promedio_previo","horas_sueno","actividades_extra","nivel_socioeconomico","acceso_internet"],
  "target_classes": { "0": 215, "1": 285 },
  "statistics": { ... },
  "correlation": { ... },
  "preview": [ { ... }, ... ]
}
```

> Guarda el `session_id` — es necesario en todos los pasos siguientes.

---

### 4. Cargar dataset desde CSV

| Campo       | Valor |
|-------------|-------|
| Método      | `POST` |
| Ruta        | `/api/v1/ia-lab/dataset/upload` |
| Content-Type| `multipart/form-data` |
| Status code | `201` |

**Form fields**

| Campo           | Tipo   | Descripción |
|-----------------|--------|-------------|
| `file`          | File   | Archivo `.csv` |
| `target_column` | string | Nombre de la columna objetivo (query param, default: `aprobado`) |

**URL de ejemplo:** `POST /api/v1/ia-lab/dataset/upload?target_column=aprobado`

**Response `201`**
```json
{
  "session_id": "3fa85f64-...",
  "message": "CSV cargado correctamente.",
  "columns": ["col1", "col2", "aprobado"],
  "total_records": 1200
}
```

---

### 5. Explorar dataset

| Campo  | Valor |
|--------|-------|
| Método | `GET` |
| Ruta   | `/api/v1/ia-lab/dataset/{session_id}/info` |

**Path params**

| Param        | Tipo   | Descripción |
|--------------|--------|-------------|
| `session_id` | string | ID de sesión obtenido al crear/cargar el dataset |

**Query params**

| Param           | Tipo   | Default    | Descripción |
|-----------------|--------|------------|-------------|
| `target_column` | string | `aprobado` | Columna objetivo |

**Response `200`** — igual al response de `/dataset/generate`

**Errors**
- `404` — sesión o columna objetivo no encontrada

---

### 6. Entrenar modelo

| Campo       | Valor |
|-------------|-------|
| Método      | `POST` |
| Ruta        | `/api/v1/ia-lab/train` |
| Status code | `201` |

**Request body** (`application/json`)
```json
{
  "session_id": "3fa85f64-...",
  "model_name": "Random Forest",
  "params": {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2
  },
  "test_size": 0.2
}
```

| Campo        | Tipo   | Rango       | Default | Descripción |
|--------------|--------|-------------|---------|-------------|
| `session_id` | string | —           | —       | ID de sesión |
| `model_name` | string | ver §2      | —       | Nombre exacto del modelo |
| `params`     | object | ver §2      | `{}`    | Hiperparámetros del modelo |
| `test_size`  | float  | 0.1 – 0.4   | `0.2`   | Fracción para test |

**Response `201`**
```json
{
  "session_id": "3fa85f64-...",
  "model_name": "Random Forest",
  "params": { "n_estimators": 100, "max_depth": 5, "min_samples_split": 2 },
  "accuracy": 0.87,
  "precision": 0.88,
  "recall": 0.86,
  "f1_score": 0.87,
  "confusion_matrix": [[95, 12], [14, 79]],
  "classification_report": { "0": { "precision": 0.87, "recall": 0.89, "f1-score": 0.88 }, "1": { ... } },
  "feature_names": ["horas_estudio", "asistencia", ...],
  "train_samples": 400,
  "test_samples": 100
}
```

**Errors**
- `404` — sesión no encontrada
- `422` — error en los datos o parámetros

---

### 7. Realizar predicción

| Campo  | Valor |
|--------|-------|
| Método | `POST` |
| Ruta   | `/api/v1/ia-lab/predict` |

**Request body** (`application/json`)
```json
{
  "session_id": "3fa85f64-...",
  "features": {
    "horas_estudio": 20.0,
    "asistencia": 85.0,
    "promedio_previo": 14.5,
    "horas_sueno": 7.0,
    "actividades_extra": 1,
    "nivel_socioeconomico": 2,
    "acceso_internet": 1
  }
}
```

| Campo        | Tipo            | Descripción |
|--------------|-----------------|-------------|
| `session_id` | string          | Sesión con modelo ya entrenado |
| `features`   | object (floats) | Mapa `{nombre_feature: valor}` |

> Los nombres de las features deben coincidir con `feature_names` del response de `/train`.

**Response `200`**
```json
{
  "session_id": "3fa85f64-...",
  "prediction": 1,
  "probabilities": { "0": 0.13, "1": 0.87 },
  "model_name": "Random Forest"
}
```

**Errors**
- `404` — sesión o modelo no encontrado
- `422` — error en los datos de entrada

---

## Resumen de rutas

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/api/v1/ia-lab/models` | Listar modelos disponibles |
| `POST` | `/api/v1/ia-lab/dataset/generate` | Generar dataset simulado |
| `POST` | `/api/v1/ia-lab/dataset/upload` | Cargar dataset CSV |
| `GET`  | `/api/v1/ia-lab/dataset/{session_id}/info` | Explorar dataset |
| `POST` | `/api/v1/ia-lab/train` | Entrenar modelo |
| `POST` | `/api/v1/ia-lab/predict` | Realizar predicción |

---

## Notas de integración

- **Sesiones en memoria:** El servidor mantiene datasets y modelos en RAM. Si el servidor se reinicia, las sesiones se pierden; el cliente debe volver a generar/cargar el dataset y reentrenar.
- **CORS:** Habilitado para todos los orígenes (`*`). Ajustar en producción.
- **Arranque local:**
  ```bash
  cd src
  uvicorn main:app --reload --port 8000
  ```
- **Dependencias necesarias:** `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `python-multipart`
