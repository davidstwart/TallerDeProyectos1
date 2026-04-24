# README_IA — Microservicio IA
## Laboratorio Interactivo de Inteligencia Artificial
### Guía completa: instalación, entorno y uso del servicio

---

## REQUISITOS PREVIOS

### 1. Instalar Python

1. Ir a https://www.python.org/downloads/
2. Descargar **Python 3.11** (recomendado) o 3.10+
3. Durante la instalación:
   - Marcar **"Add Python to PATH"**
   - Marcar **"Install pip"**
4. Verificar instalación:

```bash
py install 3.11
python --version
# Python 3.11.x

pip --version
# pip 23.x
```

### 2. Instalar Git

1. Ir a https://git-scm.com/downloads
2. Instalar con opciones por defecto
3. Verificar:

```bash
git --version
# git version 2.x.x
```

---

## CLONAR EL REPOSITORIO

```bash
git clone https://github.com/davidstwart/TallerDeProyectos1.git
cd TallerDeProyectos1
```

---

## CONFIGURAR EL ENTORNO VIRTUAL

### Windows (PowerShell)

```powershell
cd ./src/
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Si da error de permisos, ejecutar primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> Sabrás que está activo cuando veas `(venv)` al inicio de la línea de comandos.

---

## INSTALAR DEPENDENCIAS

```bash
# Actualizar pip primero (recomendado)
pip install --upgrade pip
python.exe -m pip install --upgrade pip

# Instalar dependencias con versiones fijas
pip install -r src/requirements.txt
```

**Contenido de `src/requirements.txt`:**

```
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
numpy>=1.24.0,<2.0.0
python-multipart>=0.0.6
pydantic>=2.0.0,<3.0.0
```

> Las versiones están fijadas con rangos para garantizar reproducibilidad entre entornos.
> Pydantic v2 es requerido; versiones anteriores tienen una API incompatible con FastAPI 0.100+.

Verificar instalación:

```bash
pip list
# Debe mostrar fastapi, uvicorn, pandas, scikit-learn, pydantic v2, etc.
```

---

## LEVANTAR EL SERVICIO

```bash
# Desde la raíz del repositorio
cd src
python -m uvicorn main:app --reload --host localhost --port 8000
```

> **Nota:** Usar un único proceso (sin `--workers N`) ya que el repositorio de sesiones
> vive en memoria RAM. Con múltiples workers cada proceso tendría su propio estado y
> los `session_id` no serían compartidos. Para escalar horizontalmente, reemplazar
> `InMemorySessionRepository` por un adaptador Redis.

Salida esperada:

```
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

Para detener el servidor: `Ctrl + C`

---

## VERIFICAR QUE FUNCIONA

| URL | Descripción |
|-----|-------------|
| http://localhost:8000/health | Health check → `{"status": "ok", "service": "ia-lab"}` |
| http://localhost:8000/docs | Swagger UI — probar endpoints interactivamente |
| http://localhost:8000/redoc | Documentación alternativa ReDoc |

---

## ENDPOINTS DISPONIBLES

### Sistema

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check del servicio |

### Modelos de ML

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/api/v1/ia-lab/models` | Lista modelos disponibles con sus hiperparámetros configurables |

### Dataset

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/v1/ia-lab/dataset/generate` | Genera dataset simulado de rendimiento académico |
| `POST` | `/api/v1/ia-lab/dataset/upload` | Carga dataset desde archivo CSV |
| `GET` | `/api/v1/ia-lab/dataset/{session_id}/info` | Estadísticas, correlación y preview del dataset |

### Entrenamiento y Predicción

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/v1/ia-lab/train` | Entrena un modelo ML sobre el dataset de la sesión |
| `POST` | `/api/v1/ia-lab/predict` | Realiza una predicción con el modelo entrenado |

> La referencia detallada de cada endpoint (body, response, errores) está en `src/API_ROUTES.md`.

---

## FLUJO COMPLETO DE USO

### Paso 1 — Listar modelos disponibles

```bash
curl http://localhost:8000/api/v1/ia-lab/models
```

### Paso 2 — Generar dataset simulado

```bash
curl -X POST http://localhost:8000/api/v1/ia-lab/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 500}'
```

**Respuesta:**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "total_records": 500,
  "total_features": 7,
  "feature_names": ["horas_estudio", "asistencia", "promedio_previo",
                    "horas_sueno", "actividades_extra",
                    "nivel_socioeconomico", "acceso_internet"],
  "target_classes": { "0": 215, "1": 285 }
}
```

> Guarda el `session_id` — es necesario en todos los pasos siguientes.
> Las sesiones expiran tras **1 hora de inactividad**. Genera un nuevo dataset si recibes un 404.

### Paso 2 (alternativa) — Cargar CSV propio

```bash
curl -X POST "http://localhost:8000/api/v1/ia-lab/dataset/upload?target_column=aprobado" \
  -F "file=@datos.csv"
```

**Respuesta:**
```json
{
  "session_id": "abc123...",
  "message": "CSV cargado correctamente.",
  "columns": ["col1", "col2", "aprobado"],
  "total_records": 1200
}
```

### Paso 3 — Explorar dataset

```bash
curl "http://localhost:8000/api/v1/ia-lab/dataset/3fa85f64-.../info?target_column=aprobado"
```

### Paso 4 — Entrenar modelo

```bash
curl -X POST http://localhost:8000/api/v1/ia-lab/train \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "3fa85f64-...",
    "model_name": "Random Forest",
    "params": { "n_estimators": 100, "max_depth": 5, "min_samples_split": 2 },
    "test_size": 0.2
  }'
```

**Valores permitidos para `model_name`** (exactos, con mayúsculas):

| Valor | Algoritmo |
|-------|-----------|
| `"Logistic Regression"` | Regresión Logística |
| `"Decision Tree"` | Árbol de Decisión |
| `"Random Forest"` | Bosque Aleatorio |
| `"SVM"` | Support Vector Machine |
| `"KNN"` | K-Nearest Neighbors |

**Respuesta:**
```json
{
  "session_id": "3fa85f64-...",
  "model_name": "Random Forest",
  "accuracy": 0.87,
  "precision": 0.88,
  "recall": 0.86,
  "f1_score": 0.87,
  "confusion_matrix": [[95, 12], [14, 79]],
  "feature_names": ["horas_estudio", "asistencia", "..."],
  "train_samples": 400,
  "test_samples": 100
}
```

### Paso 5 — Predecir

```bash
curl -X POST http://localhost:8000/api/v1/ia-lab/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta:**
```json
{
  "session_id": "3fa85f64-...",
  "prediction": 1,
  "probabilities": { "0": 0.13, "1": 0.87 },
  "model_name": "Random Forest"
}
```

> Los nombres en `features` deben coincidir exactamente con `feature_names`
> del response de `/train`.

---

## INTEGRACIÓN CON BACKEND PRINCIPAL (FastAPI)

```bash
pip install httpx
```

```python
import httpx

IA_URL = "http://localhost:8000/api/v1/ia-lab"

async def generar_dataset(n_samples: int = 500):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{IA_URL}/dataset/generate",
                                json={"n_samples": n_samples})
        res.raise_for_status()
        return res.json()  # contiene session_id

async def entrenar_modelo(session_id: str, model_name: str, params: dict):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{IA_URL}/train", json={
            "session_id": session_id,
            "model_name": model_name,   # ej. "Random Forest"
            "params": params,
            "test_size": 0.2,
        })
        res.raise_for_status()
        return res.json()

async def predecir(session_id: str, features: dict):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{IA_URL}/predict",
                                json={"session_id": session_id, "features": features})
        res.raise_for_status()
        return res.json()
```

---

## INTEGRACIÓN CON FRONTEND (React)

```javascript
const IA_URL = 'http://localhost:8000/api/v1/ia-lab';

// Helper con timeout de 30s
const fetchIA = async (url, options = {}) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
};

// Generar dataset simulado
export const generateDataset = (nSamples = 500) =>
  fetchIA(`${IA_URL}/dataset/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n_samples: nSamples }),
  });

// Cargar CSV propio
export const uploadDataset = (file, targetColumn = 'aprobado') => {
  const formData = new FormData();
  formData.append('file', file);
  return fetchIA(`${IA_URL}/dataset/upload?target_column=${targetColumn}`, {
    method: 'POST',
    body: formData,
  });
};

// Explorar dataset
export const getDatasetInfo = (sessionId, targetColumn = 'aprobado') =>
  fetchIA(`${IA_URL}/dataset/${sessionId}/info?target_column=${targetColumn}`);

// Listar modelos
export const getModels = () => fetchIA(`${IA_URL}/models`);

// Entrenar modelo
export const trainModel = (sessionId, modelName, params, testSize = 0.2) =>
  fetchIA(`${IA_URL}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      model_name: modelName,   // ej. "Random Forest"
      params,
      test_size: testSize,
    }),
  });

// Predecir
export const predict = (sessionId, features) =>
  fetchIA(`${IA_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, features }),
  });
```

---

## CODIGOS DE ERROR

| Código | Significado | Causa típica |
|--------|-------------|--------------|
| 404 | No encontrado | `session_id` incorrecto, sesión expirada, o sin modelo entrenado |
| 422 | Error de validación | Payload mal formado, `model_name` inválido, columna inexistente en CSV |
| 500 | Error interno | Fallo inesperado del servidor |

---

## LIMITACIONES CONOCIDAS

| Limitación | Impacto | Mitigación |
|------------|---------|------------|
| Sesiones en memoria RAM | Se pierden al reiniciar el servidor | Regenerar el dataset y reentrenar |
| Máximo 200 sesiones simultáneas | LRU: la más antigua se expulsa | Reducir `_MAX_SESSIONS` según la RAM disponible |
| TTL de 1 hora por sesión | Sesiones inactivas expiran | Reconectar y regenerar si se recibe 404 |
| Un solo proceso de uvicorn | No escala horizontalmente | Reemplazar `InMemorySessionRepository` por adaptador Redis para multi-worker |

---

## ESTRUCTURA DEL PROYECTO (rol IA)

```
src/
├── main.py                                        # Punto de entrada FastAPI + CORS + error handler
├── requirements.txt                               # Dependencias con versiones fijas
├── API_ROUTES.md                                  # Referencia completa de endpoints
│
├── domain/models/
│   └── ia_lab_model.py                            # Entidades de dominio (dataclasses)
│
├── application/
│   ├── dto/ia_lab_dto.py                          # Request/Response Pydantic con validaciones
│   ├── ports/
│   │   ├── input/ia_lab_input_port.py             # Interfaz de casos de uso (ABC)
│   │   └── output/ia_lab_output_port.py           # Interfaz de repositorio (ABC)
│   └── useCases/ia_lab_use_case.py                # Lógica de negocio ML
│
└── infrastructure/
    ├── adapters/output/session_repository.py      # Repositorio en memoria con TTL y LRU
    └── frameworks/fastapi/ia_lab_router.py        # Rutas HTTP FastAPI
```

> Los roles de backend principal y frontend también viven dentro de esta misma
> arquitectura hexagonal y son gestionados por sus respectivos equipos.

---

## CONTACTO

- **Rol:** Desarrollador IA
- **Servicio:** Microservicio de Machine Learning
- **Puerto:** `8000`
- **Arquitectura:** Hexagonal (Ports & Adapters)
