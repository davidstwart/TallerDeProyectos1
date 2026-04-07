# 🧪 Laboratorio Interactivo de Inteligencia Artificial
## Sistema Educativo - Proof of Concept (POC)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Requisitos del PMV](#2-requisitos-del-pmv)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Requisitos Previos](#4-requisitos-previos)
5. [Instalación Paso a Paso](#5-instalación-paso-a-paso)
6. [Estructura del Proyecto](#6-estructura-del-proyecto)
7. [Guía de Uso del Laboratorio](#7-guía-de-uso-del-laboratorio)
8. [Descripción Técnica del Código](#8-descripción-técnica-del-código)
9. [Modelos de Machine Learning Disponibles](#9-modelos-de-machine-learning-disponibles)
10. [Métricas de Evaluación](#10-métricas-de-evaluación)
11. [Dataset Simulado](#11-dataset-simulado)
12. [Solución de Problemas](#12-solución-de-problemas)
13. [Referencias Bibliográficas](#13-referencias-bibliográficas)

---

## 1. Descripción del Proyecto

### ¿Qué es este proyecto?

Este es un **Proof of Concept (POC)** de un sistema educativo basado en inteligencia artificial que funciona como un **laboratorio interactivo**. Permite a los estudiantes aprender de forma práctica cómo funcionan los modelos de Machine Learning mediante la experimentación directa.

### Objetivo General

Desarrollar la capacidad de desarrollo de modelos de inteligencia artificial, generando aprendizaje práctico al permitir al estudiante crear y entrenar modelos, comprendiendo la IA mediante la experiencia directa.

### Objetivos Específicos

- Permitir la **carga de datos** (simulados o propios en formato CSV).
- Ofrecer **múltiples algoritmos** de clasificación para entrenar.
- Facilitar la **modificación de hiperparámetros** en tiempo real.
- Visualizar **métricas de evaluación** y gráficas interactivas.
- Realizar **predicciones simples** con nuevos datos ingresados manualmente.
- Proporcionar una **sección educativa** que explique los conceptos clave.

---

## 2. Requisitos del PMV

| Campo | Detalle |
|-------|---------|
| **ID** | 1 |
| **PMV** | Base del sistema |
| **Meta** | La capacidad de desarrollo de modelos de inteligencia artificial |
| **Valor** | Genera aprendizaje práctico, permitiendo al estudiante crear y entrenar modelos, comprendiendo la IA mediante la experiencia directa |
| **Requerimiento** | Como estudiante, quiero cargar datos, entrenar modelos y modificar parámetros en un laboratorio interactivo, para aprender de forma práctica cómo funcionan los modelos de inteligencia artificial y realizar predicciones simples |
| **Entrega** | POC |

### Matriz de Cumplimiento

| Requerimiento | Estado | Implementación |
|---------------|--------|----------------|
| Cargar datos | ✅ Cumplido | Dataset simulado + carga de CSV propio |
| Entrenar modelos | ✅ Cumplido | 5 algoritmos: LR, DT, RF, SVM, KNN |
| Modificar parámetros | ✅ Cumplido | Panel lateral con sliders y selectores dinámicos |
| Laboratorio interactivo | ✅ Cumplido | Interfaz web con Streamlit (5 pestañas) |
| Aprender de forma práctica | ✅ Cumplido | Pestaña educativa + métricas en tiempo real |
| Predicciones simples | ✅ Cumplido | Formulario de entrada + resultado con probabilidad |

---

## 3. Arquitectura del Sistema

### Arquitectura Hexagonal

El proyecto sigue una **arquitectura hexagonal (Ports & Adapters)** que separa la lógica de negocio de la infraestructura:

```
┌─────────────────────────────────────────────────────────┐
│                 ADAPTADORES DE ENTRADA                   │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │ Streamlit │  │  CLI/Script  │  │   API REST   │     │
│   │  Web UI   │  │   Python     │  │  (FastAPI)   │     │
│   └─────┬─────┘  └──────┬───────┘  └──────┬───────┘     │
│         │               │                 │              │
│   ┌─────▼───────────────▼─────────────────▼─────┐       │
│   │           PUERTOS DE ENTRADA                 │       │
│   │  PredecirRendimiento | EntrenarModelo        │       │
│   │  ConsultarEstudiante | EvaluarModelo         │       │
│   └─────────────────┬───────────────────────┘           │
│                     │                                    │
│   ┌─────────────────▼───────────────────────┐           │
│   │          🧠 DOMINIO CORE                 │           │
│   │                                          │           │
│   │  Entidades:                              │           │
│   │    - Estudiante                          │           │
│   │    - RendimientoAcadémico                │           │
│   │    - Predicción                          │           │
│   │    - ModeloML                            │           │
│   │                                          │           │
│   │  Casos de Uso:                           │           │
│   │    - Entrenar Modelo                     │           │
│   │    - Predecir Aprobación                 │           │
│   │    - Evaluar Métricas                    │           │
│   └─────────────────┬───────────────────────┘           │
│                     │                                    │
│   ┌─────────────────▼───────────────────────┐           │
│   │           PUERTOS DE SALIDA              │           │
│   │  RepositorioEstudiante | ServicioML      │           │
│   │  ServicioVisualización | PersistModelo   │           │
│   └─────┬──────────┬──────────┬─────────┘               │
│         │          │          │                          │
│   ┌─────▼────┐ ┌───▼─────┐ ┌─▼──────────┐              │
│   │PostgreSQL│ │Scikit-  │ │ Matplotlib │              │
│   │ /CSV     │ │learn    │ │ /Seaborn   │              │
│   └──────────┘ └─────────┘ └────────────┘              │
│                 ADAPTADORES DE SALIDA                    │
└─────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```
Estudiante → Interfaz Web (Streamlit)
    → Carga/Genera datos
    → Selecciona modelo + parámetros
    → Entrena modelo (Scikit-learn)
    → Visualiza métricas (Matplotlib/Seaborn)
    → Realiza predicciones
    → Aprende conceptos
```

---

## 4. Requisitos Previos

### Software Necesario

| Software | Versión Mínima | Descarga |
|----------|---------------|----------|
| **Python** | 3.8 o superior | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | 21.0+ | Incluido con Python |
| **Git** (opcional) | 2.30+ | [git-scm.com](https://git-scm.com/) |

### Librerías de Python

| Librería | Versión Mínima | Propósito |
|----------|---------------|-----------|
| `streamlit` | 1.28+ | Interfaz web interactiva |
| `numpy` | 1.21+ | Operaciones numéricas y generación de datos |
| `pandas` | 1.3+ | Manipulación de DataFrames |
| `scikit-learn` | 1.0+ | Modelos ML, métricas, preprocesamiento |
| `matplotlib` | 3.4+ | Gráficas y visualizaciones base |
| `seaborn` | 0.11+ | Visualizaciones estadísticas avanzadas |

### Verificar si Python está instalado

Abre una terminal y ejecuta:

```bash
python --version
```

Si ves algo como `Python 3.10.12`, estás listo. Si no, descarga Python desde el enlace de arriba.

> **⚠️ IMPORTANTE (Windows):** Durante la instalación de Python, marca la casilla **"Add Python to PATH"**.

---

## 5. Instalación Paso a Paso

### Paso 1: Crear la carpeta del proyecto

**Windows (CMD o PowerShell):**
```bash
mkdir poc_educativo_ia
cd poc_educativo_ia
```

**Mac / Linux (Terminal):**
```bash
mkdir poc_educativo_ia
cd poc_educativo_ia
```

### Paso 2: Crear un entorno virtual

Un entorno virtual aísla las dependencias del proyecto para no afectar otros proyectos en tu computadora.

```bash
python -m venv venv
```

### Paso 3: Activar el entorno virtual

**Windows (CMD):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

> Si PowerShell da error de permisos, ejecuta primero:
> ```bash
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Mac / Linux:**
```bash
source venv/bin/activate
```

✅ Sabrás que está activo porque verás `(venv)` al inicio de la línea de comandos:
```
(venv) C:\Users\tu_usuario\poc_educativo_ia>
```

### Paso 4: Crear el archivo requirements.txt

Crea un archivo llamado `requirements.txt` con el siguiente contenido:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.28.0
```

### Paso 5: Instalar las dependencias

```bash
pip install -r requirements.txt
```

Esto descargará e instalará todas las librerías necesarias. Puede tardar unos minutos.

Para verificar que todo se instaló correctamente:

```bash
pip list
```

Deberías ver todas las librerías listadas con sus versiones.

### Paso 6: Crear los archivos del proyecto

Crea dos archivos Python:

#### Archivo 1: `poc_educativo.py` (Script base - evidencia para tesis)

Este es el script que se ejecuta directamente con Python y genera las gráficas como imágenes PNG. Es ideal para documentar en la tesis.

> **Nota:** El código completo de este archivo fue proporcionado en la primera entrega del POC.

#### Archivo 2: `laboratorio_ia.py` (Laboratorio interactivo - Streamlit)

Este es el laboratorio interactivo con interfaz web. Es el entregable principal del POC.

> **Nota:** El código completo de este archivo fue proporcionado en la segunda entrega del POC.

### Paso 7: Ejecutar el proyecto

#### Opción A: Script base (genera gráficas PNG)

```bash
python poc_educativo.py
```

Esto mostrará las métricas en la terminal y generará:
- `matriz_confusion.png`
- `analisis_completo.png`

#### Opción B: Laboratorio interactivo (recomendado para el POC)

```bash
streamlit run laboratorio_ia.py
```

Esto abrirá automáticamente tu navegador en `http://localhost:8501` con el laboratorio interactivo.

Para detener el servidor: presiona `Ctrl + C` en la terminal.

---

## 6. Estructura del Proyecto

```
poc_educativo_ia/
│
├── venv/                        ← Entorno virtual (NO subir a Git)
│
├── poc_educativo.py             ← Script base (evidencia tesis)
├── laboratorio_ia.py            ← Laboratorio interactivo (Streamlit)
├── requirements.txt             ← Dependencias del proyecto
├── README.md                    ← Este archivo
│
├── matriz_confusion.png         ← Generado al ejecutar poc_educativo.py
└── analisis_completo.png        ← Generado al ejecutar poc_educativo.py
```

### Archivo .gitignore (si usas Git)

Crea un archivo `.gitignore` con:

```
venv/
__pycache__/
*.pyc
.DS_Store
```

---

## 7. Guía de Uso del Laboratorio

### Pantalla Principal

Al abrir el laboratorio (`streamlit run laboratorio_ia.py`), verás:

- **Panel lateral izquierdo:** Controles de configuración
- **Área principal:** 5 pestañas con el contenido

### Panel Lateral - Configuración

#### 1. Fuente de Datos
- **Dataset simulado:** Genera datos de rendimiento académico automáticamente. Puedes ajustar el número de estudiantes (100 a 2000).
- **Cargar CSV:** Sube tu propio archivo CSV y selecciona la variable objetivo.

#### 2. Modelo de ML
Selecciona entre 5 algoritmos:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN

#### 3. Hiperparámetros
Los controles cambian dinámicamente según el modelo seleccionado:

| Modelo | Parámetros disponibles |
|--------|----------------------|
| Logistic Regression | C, máx. iteraciones, solver |
| Decision Tree | Profundidad máxima, mín. muestras, criterio |
| Random Forest | Nº árboles, profundidad máxima, mín. muestras |
| SVM | C, kernel |
| KNN | K vecinos, pesos, métrica de distancia |

#### 4. División de Datos
Ajusta el porcentaje de datos para prueba (10% a 40%).

### Pestañas del Laboratorio

| Pestaña | Contenido |
|---------|-----------|
| 📋 **Explorar Datos** | Vista previa, estadísticas, distribución, correlaciones |
| 🏋️ **Entrenar Modelo** | Botón de entrenamiento, coeficientes, importancia de variables |
| 📈 **Métricas y Evaluación** | Accuracy, Precision, Recall, F1, matriz de confusión |
| 🔮 **Realizar Predicción** | Formulario para predecir con datos nuevos |
| 📚 **Aprendizaje** | Explicación de conceptos, tabla comparativa de modelos |

### Flujo de Trabajo Recomendado

1. **Selecciona** la fuente de datos en el panel lateral.
2. **Explora** los datos en la primera pestaña.
3. **Elige** un modelo y ajusta los hiperparámetros.
4. **Entrena** el modelo haciendo clic en el botón "ENTRENAR MODELO".
5. **Evalúa** las métricas en la tercera pestaña.
6. **Experimenta** cambiando parámetros y comparando resultados.
7. **Predice** con nuevos datos en la cuarta pestaña.
8. **Aprende** los conceptos en la quinta pestaña.

---

## 8. Descripción Técnica del Código

### 8.1 Generación del Dataset Simulado

```python
def generar_dataset_simulado(n=500):
```

Se generan **7 variables independientes** que simulan factores que influyen en el rendimiento académico:

| Variable | Rango | Tipo | Justificación |
|----------|-------|------|---------------|
| `horas_estudio` | 0 - 30 | Continua | Horas de estudio semanales |
| `asistencia` | 40% - 100% | Continua | Porcentaje de asistencia a clases |
| `promedio_previo` | 5 - 20 | Continua | Promedio de notas previas (escala vigesimal) |
| `horas_sueno` | 3 - 10 | Continua | Horas de sueño diarias |
| `actividades_extra` | 0 / 1 | Binaria | Participación en actividades extracurriculares |
| `nivel_socioeconomico` | 1 / 2 / 3 | Categórica | Bajo / Medio / Alto |
| `acceso_internet` | 0 / 1 | Binaria | Acceso a internet en casa |

La **variable objetivo** (`aprobado`) se calcula mediante una función ponderada:

```
puntaje = 0.25 × (horas_estudio/30)
        + 0.20 × (asistencia/100)
        + 0.25 × (promedio_previo/20)
        + 0.10 × (horas_sueno/10)
        + 0.08 × actividades_extra
        + 0.07 × (nivel_socioeconomico/3)
        + 0.05 × acceso_internet
        + ruido_gaussiano(0, 0.05)

aprobado = 1 si puntaje ≥ 0.55, sino 0
```

### 8.2 Preprocesamiento

1. **Separación de variables:** `X` (features) e `y` (target).
2. **División train/test:** Usando `train_test_split` con `stratify` para mantener la proporción de clases.
3. **Estandarización:** `StandardScaler` ajustado **solo** con datos de entrenamiento (evita data leakage).

### 8.3 Entrenamiento

El modelo seleccionado se instancia con los hiperparámetros configurados por el usuario y se entrena con `modelo.fit(X_train_scaled, y_train)`.

### 8.4 Evaluación

Se calculan 4 métricas principales y se genera una matriz de confusión visual (absoluta y normalizada).

### 8.5 Predicción

Los nuevos datos ingresados se escalan con el **mismo scaler** del entrenamiento antes de pasar al modelo.

---

## 9. Modelos de Machine Learning Disponibles

### Logistic Regression (Regresión Logística)

- **Tipo:** Clasificación lineal
- **Cómo funciona:** Modela la probabilidad de pertenencia a una clase usando la función sigmoide.
- **Ventajas:** Interpretable, rápido, funciona bien con datos linealmente separables.
- **Hiperparámetros clave:**
  - `C`: Inverso de la fuerza de regularización. Valores altos = menos regularización.
  - `solver`: Algoritmo de optimización (lbfgs, liblinear, etc.).

### Decision Tree (Árbol de Decisión)

- **Tipo:** Clasificación no lineal
- **Cómo funciona:** Crea reglas de decisión tipo "si X > umbral, entonces clase A".
- **Ventajas:** Muy interpretable, no requiere escalado.
- **Hiperparámetros clave:**
  - `max_depth`: Profundidad máxima del árbol. Controla el overfitting.
  - `criterion`: Función para medir la calidad de la división (gini o entropy).

### Random Forest (Bosque Aleatorio)

- **Tipo:** Ensemble de árboles de decisión
- **Cómo funciona:** Entrena múltiples árboles con subconjuntos aleatorios y vota por la clase mayoritaria.
- **Ventajas:** Robusto, reduce overfitting, maneja bien datos complejos.
- **Hiperparámetros clave:**
  - `n_estimators`: Número de árboles en el bosque.
  - `max_depth`: Profundidad máxima de cada árbol.

### SVM (Support Vector Machine)

- **Tipo:** Clasificación basada en márgenes
- **Cómo funciona:** Busca el hiperplano que maximiza el margen entre clases.
- **Ventajas:** Efectivo en espacios de alta dimensión.
- **Hiperparámetros clave:**
  - `C`: Parámetro de regularización.
  - `kernel`: Función kernel (rbf, linear, poly, sigmoid).

### KNN (K-Nearest Neighbors)

- **Tipo:** Clasificación basada en instancias
- **Cómo funciona:** Clasifica según la clase mayoritaria de los K vecinos más cercanos.
- **Ventajas:** Simple, no requiere entrenamiento explícito.
- **Hiperparámetros clave:**
  - `n_neighbors`: Número de vecinos a considerar.
  - `weights`: Ponderación (uniform o distance).

---

## 10. Métricas de Evaluación

### Accuracy (Exactitud)

```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
```

¿Qué porcentaje del total de predicciones fueron correctas?

### Precision (Precisión)

```
Precision = VP / (VP + FP)
```

De todos los que el modelo predijo como "aprobado", ¿cuántos realmente lo eran?

### Recall (Sensibilidad)

```
Recall = VP / (VP + FN)
```

De todos los estudiantes que realmente aprobaron, ¿cuántos detectó el modelo?

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Media armónica entre Precision y Recall. Útil cuando las clases están desbalanceadas.

### Matriz de Confusión

```
                    Predicción
                 Negativo  Positivo
Real  Negativo [   VN    |   FP   ]
      Positivo [   FN    |   VP   ]
```

Donde:
- **VP (Verdadero Positivo):** Predijo aprobado y era aprobado ✅
- **VN (Verdadero Negativo):** Predijo desaprobado y era desaprobado ✅
- **FP (Falso Positivo):** Predijo aprobado pero era desaprobado ❌
- **FN (Falso Negativo):** Predijo desaprobado pero era aprobado ❌

---

## 11. Dataset Simulado

### Descripción de Variables

| # | Variable | Tipo | Rango | Descripción |
|---|----------|------|-------|-------------|
| 1 | `horas_estudio` | Float | 0.0 - 30.0 | Horas de estudio semanales |
| 2 | `asistencia` | Float | 40.0 - 100.0 | Porcentaje de asistencia a clases |
| 3 | `promedio_previo` | Float | 5.0 - 20.0 | Promedio de notas previas |
| 4 | `horas_sueno` | Float | 3.0 - 10.0 | Horas de sueño diarias |
| 5 | `actividades_extra` | Int | 0 / 1 | 0=No, 1=Sí |
| 6 | `nivel_socioeconomico` | Int | 1 / 2 / 3 | 1=Bajo, 2=Medio, 3=Alto |
| 7 | `acceso_internet` | Int | 0 / 1 | 0=No, 1=Sí |
| 8 | `aprobado` | Int | 0 / 1 | **Variable objetivo** (0=Desaprobado, 1=Aprobado) |

### Pesos de la Variable Objetivo

| Variable | Peso | Justificación |
|----------|------|---------------|
| Horas de estudio | 25% | Factor principal de rendimiento |
| Promedio previo | 25% | Indicador histórico de capacidad |
| Asistencia | 20% | Correlación directa con aprendizaje |
| Horas de sueño | 10% | Impacto en concentración y memoria |
| Actividades extra | 8% | Desarrollo integral del estudiante |
| Nivel socioeconómico | 7% | Acceso a recursos educativos |
| Acceso a internet | 5% | Herramienta de apoyo al estudio |

---

## 12. Solución de Problemas

### Error: "python no se reconoce como comando"

**Causa:** Python no está en el PATH del sistema.

**Solución Windows:**
1. Reinstala Python marcando "Add Python to PATH".
2. O usa `python3` en lugar de `python`.

**Solución Mac/Linux:**
```bash
python3 --version
# Si funciona, usa python3 en todos los comandos
```

### Error: "No module named streamlit"

**Causa:** Las dependencias no están instaladas o el entorno virtual no está activo.

**Solución:**
```bash
# Activa el entorno virtual primero
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Luego instala:
pip install -r requirements.txt
```

### Error: "Address already in use" al ejecutar Streamlit

**Causa:** Otra instancia de Streamlit está corriendo.

**Solución:**
```bash
# Usa otro puerto
streamlit run laboratorio_ia.py --server.port 8502
```

### Error de permisos en PowerShell

**Solución:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Las gráficas no se muestran

**Causa:** Falta `matplotlib` o hay conflicto de backend.

**Solución:**
```bash
pip install matplotlib --upgrade
```

### El modelo no se entrena (botón no responde)

**Causa:** Puede ser un problema de caché de Streamlit.

**Solución:**
1. Presiona `Ctrl + C` en la terminal.
2. Ejecuta de nuevo: `streamlit run laboratorio_ia.py`
3. En el navegador, presiona `Ctrl + Shift + R` (hard refresh).

---

## 13. Referencias Bibliográficas

### Librerías y Frameworks

1. **Scikit-learn:** Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830. [scikit-learn.org](https://scikit-learn.org/)

2. **Streamlit:** Streamlit Inc. (2019). *Streamlit — The fastest way to build data apps*. [streamlit.io](https://streamlit.io/)

3. **Pandas:** McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference. [pandas.pydata.org](https://pandas.pydata.org/)

4. **NumPy:** Harris, C.R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357-362. [numpy.org](https://numpy.org/)

5. **Matplotlib:** Hunter, J.D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90-95. [matplotlib.org](https://matplotlib.org/)

6. **Seaborn:** Waskom, M. (2021). *seaborn: statistical data visualization*. Journal of Open Source Software, 6(60), 3021. [seaborn.pydata.org](https://seaborn.pydata.org/)

### Conceptos de Machine Learning

7. **Regresión Logística:** Hosmer, D.W., Lemeshow, S., & Sturdivant, R.X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

8. **Árboles de Decisión:** Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.

9. **Random Forest:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

10. **SVM:** Cortes, C. & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273-297.

11. **KNN:** Cover, T. & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21-27.

### Educación e IA

12. **Rendimiento Académico y ML:** Shahiri, A.M., Husain, W., & Rashid, N.A. (2015). *A Review on Predicting Student's Performance Using Data Mining Techniques*. Procedia Computer Science, 72, 414-422.

13. **Arquitectura Hexagonal:** Cockburn, A. (2005). *Hexagonal Architecture*. [alistair.cockburn.us](https://alistair.cockburn.us/hexagonal-architecture/)

---

## Licencia

Este proyecto es un **Proof of Concept** desarrollado con fines académicos y de investigación.

---

> **Desarrollado como parte del PMV 1 - Base del Sistema**
> **Entrega: POC - Laboratorio Interactivo de Inteligencia Artificial**
