# 🧠 IA_INTEGRATION.md — Microservicio IA
## Guía completa: desde instalación hasta levantar el servicio

---

## 📋 REQUISITOS PREVIOS

### 1. Instalar Python

1. Ir a https://www.python.org/downloads/
2. Descargar **Python 3.11** (recomendado) o 3.10+
3. Durante la instalación:
   - ✅ Marcar **"Add Python to PATH"**
   - ✅ Marcar **"Install pip"**
4. Verificar instalación:

```bash
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

## 📥 CLONAR EL REPOSITORIO

```bash
# Clonar el proyecto
git clone https://github.com/davidstwart/TallerDeProyectos1.git

# Entrar a la carpeta del microservicio IA
cd .\TallerDeProyectos1\
```

> ⚠️ Reemplaza `<organización>/<repositorio>` con la URL real del repo.

---

## ⚙️ CONFIGURAR EL ENTORNO VIRTUAL

### Windows (PowerShell)

```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1

# Si da error de permisos, ejecutar primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux / macOS

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

> ✅ Sabrás que está activo cuando veas `(venv)` al inicio de la línea en la terminal.

---

## 📦 INSTALAR DEPENDENCIAS

```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**

```
fastapi
uvicorn[standard]
pandas
scikit-learn
numpy
python-multipart
pydantic
```

> ⚠️ No uses versiones fijas (como `pandas==2.2.3`) para evitar errores de compilación en Windows.

Verificar instalación:

```bash
pip list
# Debe mostrar fastapi, uvicorn, pandas, scikit-learn, etc.
```

---

## 🚀 LEVANTAR EL SERVICIO

```bash
# Desde la carpeta raíz del microservicio (donde está /src)
cd src
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Deberías ver:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

---

## 🌐 VERIFICAR QUE FUNCIONA

Abrir en el navegador:

| URL | Descripción |
|-----|-------------|
| http://localhost:8000/ | Info del servicio |
| http://localhost:8000/health | Health check → `{"status": "ok"}` |
| http://localhost:8000/api/v1/docs | **Swagger UI** (probar endpoints) |
| http://localhost:8000/api/v1/redoc | Documentación alternativa |

---

## 🛤️ ENDPOINTS DISPONIBLES

### 📂 Datasets

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/ai/datasets/upload` | Sube CSV (multipart/form-data) |
| GET | `/api/v1/ai/datasets/{dataset_id}` | Info del dataset cargado |
| DELETE | `/api/v1/ai/datasets/{dataset_id}` | Elimina dataset de memoria |

### 🤖 Modelos

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/ai/models/train` | Entrena un modelo ML |
| GET | `/api/v1/ai/models/{model_id}/metrics` | Métricas detalladas |
| POST | `/api/v1/ai/models/{model_id}/predict` | Predicción individual |
| DELETE | `/api/v1/ai/models/{model_id}` | Elimina modelo de memoria |

### 🏥 Sistema

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Info del servicio |
| GET | `/health` | Health check |

---

## 📦 FLUJO COMPLETO DE USO

### Paso 1 — Subir dataset

```bash
curl -X POST http://localhost:8000/api/v1/ai/datasets/upload \
  -F "file=@datos.csv"
```

**Respuesta:**
```json
{
  "message": "Dataset cargado exitosamente",
  "dataset_id": "uuid-123",
  "filename": "datos.csv",
  "columns": ["horas_estudio", "asistencia", "aprobado"],
  "numeric_columns": ["horas_estudio", "asistencia"],
  "row_count": 500
}
```

### Paso 2 — Entrenar modelo

```bash
curl -X POST http://localhost:8000/api/v1/ai/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-123",
    "target_column": "aprobado",
    "model_type": "random_forest",
    "test_size": 0.2,
    "params": {"n_estimators": 100, "max_depth": 5}
  }'
```

**Modelos disponibles:** `logistic_regression`, `decision_tree`, `random_forest`, `svm`, `knn`

**Respuesta:**
```json
{
  "message": "Modelo entrenado exitosamente",
  "model_id": "uuid-456",
  "model_type": "random_forest",
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1_score": 0.85
  },
  "feature_names": ["horas_estudio", "asistencia"],
  "class_labels": ["0", "1"]
}
```

### Paso 3 — Ver métricas

```bash
curl http://localhost:8000/api/v1/ai/models/uuid-456/metrics
```

### Paso 4 — Predecir

```bash
curl -X POST http://localhost:8000/api/v1/ai/models/uuid-456/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "horas_estudio": 20.0,
      "asistencia": 90.0
    }
  }'
```

**Respuesta:**
```json
{
  "prediction": 1,
  "label": "1",
  "probabilities": {"0": 0.15, "1": 0.85},
  "model_id": "uuid-456"
}
```

---

## 🔗 INTEGRACIÓN CON BACKEND PRINCIPAL (FastAPI)

Instalar cliente HTTP:
```bash
pip install httpx
```

```python
import httpx

IA_URL = "http://localhost:8000/api/v1"

async def entrenar_modelo(dataset_id: str, target: str, model_type: str = "random_forest"):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{IA_URL}/ai/models/train", json={
            "dataset_id": dataset_id,
            "target_column": target,
            "model_type": model_type,
            "test_size": 0.2,
            "params": {}
        })
        res.raise_for_status()
        return res.json()

async def predecir(model_id: str, features: dict):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{IA_URL}/ai/models/{model_id}/predict",
            json={"features": features}
        )
        res.raise_for_status()
        return res.json()
```

---

## 🔗 INTEGRACIÓN CON FRONTEND (React)

```javascript
const IA_URL = 'http://localhost:8000/api/v1';

// Subir dataset
export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${IA_URL}/ai/datasets/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

// Entrenar modelo
export const trainModel = async (config) => {
  const res = await fetch(`${IA_URL}/ai/models/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

// Predecir
export const predict = async (modelId, features) => {
  const res = await fetch(`${IA_URL}/ai/models/${modelId}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};
```

---

## ⚠️ CÓDIGOS DE ERROR

| Código | Significado | Ejemplo |
|--------|-------------|---------|
| 400 | Datos inválidos | CSV vacío, columna no existe |
| 404 | No encontrado | model_id o dataset_id incorrecto |
| 422 | Error de validación | Payload mal formado |
| 500 | Error interno | Fallo inesperado del servidor |

---

## 🧪 EJECUTAR TESTS DE INTEGRACIÓN

```bash
# Desde la raíz del proyecto (con el servicio ya levantado)
python src/tests/integration/test_api_integration.py
```

Salida esperada:
```
==================================================
🧪 TEST DE INTEGRACIÓN - MICROSERVICIO IA
==================================================
✅ 1/5 Health check OK
✅ 2/5 Dataset subido: uuid-xxx
✅ 3/5 Modelo entrenado: uuid-yyy  →  Accuracy: 0.85
✅ 4/5 Métricas obtenidas: F1=0.84
✅ 5/5 Predicción: 1 (prob: {"0": 0.15, "1": 0.85})

==================================================
🎉 TODOS LOS TESTS PASARON EXITOSAMENTE
==================================================
```

---

## 📁 ESTRUCTURA DEL PROYECTO

```
src/
├── domain/models/          # Entidades de dominio
├── application/
│   ├── dto/                # Objetos de transferencia
│   ├── ports/              # Interfaces (input/output)
│   └── use_cases/          # Lógica de negocio IA
├── infrastructure/
│   ├── adapters/output/    # Repositorios in-memory
│   ├── config/             # Settings y CORS
│   └── frameworks/fastapi/ # Routers y app FastAPI
├── tests/integration/      # Tests de integración
└── main.py                 # Punto de entrada
```

---

## 👤 CONTACTO

- **Rol:** Desarrollador IA
- **Servicio:** Microservicio de Machine Learning
- **Puerto:** `8000`
- **Arquitectura:** Hexagonal
