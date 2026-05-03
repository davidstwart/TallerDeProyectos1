# README_IA — Microservicio IA

## Laboratorio Interactivo de Inteligencia Artificial

### Guía completa: instalación, entorno y uso del servicio

---

## REQUISITOS PREVIOS

### 1. Instalar Python

1. Ir a [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Descargar **Python 3.11** (recomendado)
3. Durante la instalación:

   * Marcar **"Add Python to PATH"**
   * Marcar **"Install pip"**
4. Verificar instalación:

```bash
python --version
pip --version
```

---

## CLONAR EL REPOSITORIO

```bash
git clone https://github.com/davidstwart/TallerDeProyectos1.git
cd TallerDeProyectos1
```

---

## CONFIGURAR EL ENTORNO VIRTUAL

```bash
cd ./src/
py -3.11 -m venv venv
venv\Scripts\activate
```

---

## INSTALAR DEPENDENCIAS

```bash
pip install --upgrade pip
python.exe -m pip install --upgrade pip


pip install -r requirements.txt
```

### requirements.txt

```txt
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
numpy>=1.24.0,<2.0.0
python-multipart>=0.0.6
pydantic>=2.0.0,<3.0.0
joblib>=1.3.0,<2.0.0
```

---

## LEVANTAR EL SERVICIO

```bash
cd src
python -m uvicorn main:app --reload --host localhost --port 8000
```

---

## ENDPOINTS DISPONIBLES

| Método | Ruta                                     | Descripción              |
| ------ | ---------------------------------------- | ------------------------ |
| GET    | /api/v1/ia-lab/models                    | Lista modelos            |
| POST   | /api/v1/ia-lab/dataset/generate          | Generar dataset          |
| POST   | /api/v1/ia-lab/dataset/upload            | Subir CSV                |
| GET    | /api/v1/ia-lab/dataset/{session_id}/info | Explorar dataset         |
| POST   | /api/v1/ia-lab/train                     | Entrenar modelo          |
| POST   | /api/v1/ia-lab/predict                   | Predecir                 |
| POST   | /api/v1/ia-lab/model/load                | Cargar modelo persistido |

---

# 🚀 FLUJO COMPLETO (POSTMAN)

---

## 🔹 1. Listar modelos

* Método: GET
* URL:

```
http://localhost:8000/api/v1/ia-lab/models
```

---

## 🔹 2. Generar dataset

* Método: POST
* URL:

```
http://localhost:8000/api/v1/ia-lab/dataset/generate
```

* Body (JSON):

```json
{
  "n_samples": 500
}
```

---

## 🔹 3. Entrenar modelo

* Método: POST
* URL:

```
http://localhost:8000/api/v1/ia-lab/train
```

* Body:

```json
{
  "session_id": "{{session_id}}",
  "model_name": "Random Forest",
  "params": {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2
  },
  "test_size": 0.2
}
```

---

## 🔹 4. Predecir

* Método: POST
* URL:

```
http://localhost:8000/api/v1/ia-lab/predict
```

* Body:

```json
{
  "session_id": "{{session_id}}",
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

---

# 🔁 REUTILIZAR MODELOS (NUEVO)

## 🔹 5. Cargar modelo existente

* Método: POST
* URL:

```
http://localhost:8000/api/v1/ia-lab/model/load
```

* Body:

```json
{
  "model_id": "random_forest_xxx"
}
```

---

## 🔹 6. Predecir con modelo cargado

* Método: POST
* URL:

```
http://localhost:8000/api/v1/ia-lab/predict
```

* Body:

```json
{
  "session_id": "{{session_id}}",
  "features": {
    "horas_estudio": 18.0,
    "asistencia": 90.0,
    "promedio_previo": 15.2,
    "horas_sueno": 6.5,
    "actividades_extra": 0,
    "nivel_socioeconomico": 2,
    "acceso_internet": 1
  }
}
```

---

# 🧠 ARQUITECTURA DE MODELOS

* Entrenamiento → memoria
* Persistencia → `/models/*.pkl`
* Carga → desde disco
* Inferencia → desde sesión

---

# ⚠️ CONSIDERACIONES

* Las sesiones expiran en 1 hora
* Los modelos persistidos sobreviven reinicios
* Las features deben coincidir exactamente

---

# 🧱 ESTRUCTURA

```
src/
├── domain/
├── application/
├── infrastructure/
│   ├── adapters/output/session_repository.py  # memoria + joblib
│   └── frameworks/fastapi/ia_lab_router.py
```

---

# 🎯 ESTADO

✔️ Entrenamiento
✔️ Evaluación
✔️ Persistencia
✔️ Reutilización
✔️ Inferencia desacoplada

👉 Sistema listo para uso tipo MLOps básico

---

# 🔗 INTEGRACIÓN CON BACKEND (FastAPI / Python)

Instalar cliente HTTP:

```bash
pip install httpx
```

Ejemplo de integración asíncrona:

```python
import httpx

IA_URL = "http://localhost:8000/api/v1/ia-lab"

async def generar_dataset(n_samples: int = 500):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{IA_URL}/dataset/generate",
            json={"n_samples": n_samples}
        )
        res.raise_for_status()
        return res.json()  # contiene session_id


async def entrenar_modelo(session_id: str, model_name: str, params: dict, test_size: float = 0.2):
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            f"{IA_URL}/train",
            json={
                "session_id": session_id,
                "model_name": model_name,
                "params": params,
                "test_size": test_size,
            },
        )
        res.raise_for_status()
        return res.json()


async def cargar_modelo(model_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{IA_URL}/model/load",
            json={"model_id": model_id},
        )
        res.raise_for_status()
        return res.json()  # contiene session_id


async def predecir(session_id: str, features: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{IA_URL}/predict",
            json={"session_id": session_id, "features": features},
        )
        res.raise_for_status()
        return res.json()
```

Notas:

* Maneja timeouts y errores con `raise_for_status()`.
* Puedes cachear `session_id` en tu backend para evitar recrearlo.

---

# 🌐 INTEGRACIÓN CON FRONTEND (React)

```javascript
const IA_URL = 'http://localhost:8000/api/v1/ia-lab';

// Helper con timeout
export const fetchIA = async (url, options = {}) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Request failed');
    }
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
};

// 1) Generar dataset
export const generateDataset = (nSamples = 500) =>
  fetchIA(`${IA_URL}/dataset/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n_samples: nSamples }),
  });

// 2) Entrenar modelo
export const trainModel = (sessionId, modelName, params, testSize = 0.2) =>
  fetchIA(`${IA_URL}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      model_name: modelName,
      params,
      test_size: testSize,
    }),
  });

// 3) Cargar modelo persistido
export const loadModel = (modelId) =>
  fetchIA(`${IA_URL}/model/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });

// 4) Predecir
export const predict = (sessionId, features) =>
  fetchIA(`${IA_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, features }),
  });
```

Buenas prácticas en frontend:

* Persistir `session_id` en estado global (Redux/Zustand/Context).
* Validar que las `features` coincidan con `feature_names`.
* Mostrar feedback de carga y manejo de errores.

---

# 🧠 PATRÓN DE USO RECOMENDADO

1. Generar dataset → guardar `session_id`
2. Entrenar modelo → (opcional) persistir en backend
3. Predecir

Ó

1. Cargar modelo (`model_id`) → nuevo `session_id`
2. Predecir sin re-entrenar

---

# ⚠️ ERRORES COMUNES

* 404: `session_id` expirado o inexistente
* 422: payload inválido o `model_name` incorrecto
* Predicción incorrecta: mismatch de features

---

# 🧪 CHECK RÁPIDO

* `/models` responde
* `/dataset/generate` devuelve `session_id`
* `/train` devuelve métricas
* `/predict` devuelve resultado
* `/model/load` permite reutilizar modelo

---

# 🚀 SIGUIENTE NIVEL

* Versionado de modelos (model_id + version)
* Almacenamiento en S3 / MinIO
* Tracking con MLflow
* Cache distribuido (Redis)
