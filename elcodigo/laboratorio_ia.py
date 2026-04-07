# ============================================================================
# POC: LABORATORIO INTERACTIVO DE INTELIGENCIA ARTIFICIAL
# Sistema Educativo - Base del Sistema (PMV 1)
# ============================================================================
# Meta: Desarrollar la capacidad de desarrollo de modelos de IA
# Valor: Genera aprendizaje práctico, permitiendo al estudiante crear y
#        entrenar modelos, comprendiendo la IA mediante experiencia directa.
# Entrega: POC
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import io

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="🧪 Laboratorio IA - Sistema Educativo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Laboratorio Interactivo de Inteligencia Artificial")
st.markdown("""
> **Objetivo:** Cargar datos, entrenar modelos y modificar parámetros para aprender
> de forma práctica cómo funcionan los modelos de IA y realizar predicciones simples.
""")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_dataset_simulado(n=500):
    """Genera un dataset simulado de rendimiento académico."""
    np.random.seed(42)
    datos = {
        'horas_estudio': np.round(np.random.uniform(0, 30, n), 1),
        'asistencia': np.round(np.random.uniform(40, 100, n), 1),
        'promedio_previo': np.round(np.random.uniform(5, 20, n), 1),
        'horas_sueno': np.round(np.random.uniform(3, 10, n), 1),
        'actividades_extra': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'nivel_socioeconomico': np.random.choice([1, 2, 3], n, p=[0.3, 0.5, 0.2]),
        'acceso_internet': np.random.choice([0, 1], n, p=[0.2, 0.8]),
    }
    df = pd.DataFrame(datos)
    puntaje = (
        0.25 * df['horas_estudio'] / 30 +
        0.20 * df['asistencia'] / 100 +
        0.25 * df['promedio_previo'] / 20 +
        0.10 * df['horas_sueno'] / 10 +
        0.08 * df['actividades_extra'] +
        0.07 * df['nivel_socioeconomico'] / 3 +
        0.05 * df['acceso_internet']
    )
    ruido = np.random.normal(0, 0.05, n)
    df['aprobado'] = (puntaje + ruido >= 0.55).astype(int)
    return df


def obtener_modelo(nombre, params):
    """Retorna el modelo seleccionado con los parámetros configurados."""
    if nombre == "Logistic Regression":
        return LogisticRegression(
            C=params['C'],
            max_iter=params['max_iter'],
            solver=params['solver'],
            random_state=42
        )
    elif nombre == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            criterion=params['criterion'],
            random_state=42
        )
    elif nombre == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42
        )
    elif nombre == "SVM":
        return SVC(
            C=params['C'],
            kernel=params['kernel'],
            probability=True,
            random_state=42
        )
    elif nombre == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            metric=params['metric']
        )


# ============================================================================
# SIDEBAR: CONFIGURACIÓN DEL LABORATORIO
# ============================================================================
st.sidebar.header("⚙️ Panel de Control")

# --- PASO 1: CARGA DE DATOS ---
st.sidebar.subheader("📂 1. Fuente de Datos")
fuente_datos = st.sidebar.radio(
    "Selecciona la fuente:",
    ["Dataset simulado (Rendimiento Académico)", "Cargar mi propio CSV"]
)

df = None

if fuente_datos == "Dataset simulado (Rendimiento Académico)":
    n_muestras = st.sidebar.slider("Número de estudiantes:", 100, 2000, 500, 50)
    df = generar_dataset_simulado(n_muestras)
    variable_objetivo = 'aprobado'
else:
    archivo = st.sidebar.file_uploader("Sube tu archivo CSV:", type=['csv'])
    if archivo is not None:
        df = pd.read_csv(archivo)
        variable_objetivo = st.sidebar.selectbox(
            "Selecciona la variable objetivo (binaria):",
            df.columns.tolist()
        )

# --- PASO 2: SELECCIÓN DE MODELO ---
st.sidebar.subheader("🤖 2. Modelo de ML")
modelo_nombre = st.sidebar.selectbox(
    "Selecciona el algoritmo:",
    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]
)

# --- PASO 3: HIPERPARÁMETROS ---
st.sidebar.subheader("🎛️ 3. Hiperparámetros")
params = {}

if modelo_nombre == "Logistic Regression":
    params['C'] = st.sidebar.slider("C (Regularización inversa):", 0.01, 10.0, 1.0, 0.01)
    params['max_iter'] = st.sidebar.slider("Máx. iteraciones:", 100, 2000, 1000, 100)
    params['solver'] = st.sidebar.selectbox("Solver:", ['lbfgs', 'liblinear', 'newton-cg', 'saga'])

elif modelo_nombre == "Decision Tree":
    params['max_depth'] = st.sidebar.slider("Profundidad máxima:", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.slider("Mín. muestras para dividir:", 2, 20, 2)
    params['criterion'] = st.sidebar.selectbox("Criterio:", ['gini', 'entropy'])

elif modelo_nombre == "Random Forest":
    params['n_estimators'] = st.sidebar.slider("Número de árboles:", 10, 300, 100, 10)
    params['max_depth'] = st.sidebar.slider("Profundidad máxima:", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.slider("Mín. muestras para dividir:", 2, 20, 2)

elif modelo_nombre == "SVM":
    params['C'] = st.sidebar.slider("C (Regularización):", 0.01, 10.0, 1.0, 0.01)
    params['kernel'] = st.sidebar.selectbox("Kernel:", ['rbf', 'linear', 'poly', 'sigmoid'])

elif modelo_nombre == "KNN":
    params['n_neighbors'] = st.sidebar.slider("K (vecinos):", 1, 21, 5, 2)
    params['weights'] = st.sidebar.selectbox("Pesos:", ['uniform', 'distance'])
    params['metric'] = st.sidebar.selectbox("Métrica:", ['euclidean', 'manhattan', 'minkowski'])

# --- PASO 4: DIVISIÓN DE DATOS ---
st.sidebar.subheader("📊 4. División de Datos")
test_size = st.sidebar.slider("Porcentaje de prueba (%):", 10, 40, 20, 5) / 100

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================

if df is not None:

    # --- TAB LAYOUT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Explorar Datos",
        "🏋️ Entrenar Modelo",
        "📈 Métricas y Evaluación",
        "🔮 Realizar Predicción",
        "📚 Aprendizaje"
    ])

    # ==========================
    # TAB 1: EXPLORAR DATOS
    # ==========================
    with tab1:
        st.header("📋 Exploración del Dataset")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total registros", df.shape[0])
        col2.metric("Variables", df.shape[1])
        col3.metric("Aprobados", f"{df[variable_objetivo].mean()*100:.1f}%")
        col4.metric("Desaprobados", f"{(1-df[variable_objetivo].mean())*100:.1f}%")

        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Distribución de la variable objetivo")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors = ['#e74c3c', '#2ecc71']
        df[variable_objetivo].value_counts().plot(kind='bar', ax=axes[0], color=colors)
        axes[0].set_title('Conteo')
        axes[0].set_xticklabels(['Desaprobado (0)', 'Aprobado (1)'], rotation=0)
        df[variable_objetivo].value_counts().plot(kind='pie', ax=axes[1], colors=colors,
            autopct='%1.1f%%', labels=['Desaprobado', 'Aprobado'])
        axes[1].set_title('Proporción')
        axes[1].set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Matriz de correlación")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax2, linewidths=0.5)
        ax2.set_title('Correlación entre variables')
        plt.tight_layout()
        st.pyplot(fig2)

    # ==========================
    # TAB 2: ENTRENAR MODELO
    # ==========================
    with tab2:
        st.header("🏋️ Entrenamiento del Modelo")

        # Preparación de datos
        X = df.drop(variable_objetivo, axis=1)
        y = df[variable_objetivo]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Mostrar configuración
        st.info(f"""
        **Configuración actual:**
        - 🤖 Modelo: **{modelo_nombre}**
        - 📊 Train: **{X_train.shape[0]}** muestras | Test: **{X_test.shape[0]}** muestras
        - 🎛️ Parámetros: {params}
        """)

        if st.button("🚀 ENTRENAR MODELO", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo..."):
                modelo = obtener_modelo(modelo_nombre, params)
                modelo.fit(X_train_scaled, y_train)

                # Guardar en session_state
                st.session_state['modelo'] = modelo
                st.session_state['scaler'] = scaler
                st.session_state['X_test_scaled'] = X_test_scaled
                st.session_state['y_test'] = y_test
                st.session_state['X_train'] = X_train
                st.session_state['X'] = X
                st.session_state['modelo_nombre'] = modelo_nombre
                st.session_state['params'] = params
                st.session_state['entrenado'] = True

            st.success(f"✅ Modelo **{modelo_nombre}** entrenado exitosamente!")

            # Mostrar coeficientes si aplica
            if hasattr(modelo, 'coef_'):
                st.subheader("Coeficientes del modelo")
                coef_df = pd.DataFrame({
                    'Variable': X.columns,
                    'Coeficiente': modelo.coef_[0]
                }).sort_values('Coeficiente', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(coef_df['Variable'], coef_df['Coeficiente'],
                    color=['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coeficiente']])
                ax.set_title(f'Importancia de Variables - {modelo_nombre}')
                ax.axvline(x=0, color='black', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)

            elif hasattr(modelo, 'feature_importances_'):
                st.subheader("Importancia de variables")
                imp_df = pd.DataFrame({
                    'Variable': X.columns,
                    'Importancia': modelo.feature_importances_
                }).sort_values('Importancia', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(imp_df['Variable'], imp_df['Importancia'], color='#3498db')
                ax.set_title(f'Importancia de Variables - {modelo_nombre}')
                plt.tight_layout()
                st.pyplot(fig)

    # ==========================
    # TAB 3: MÉTRICAS
    # ==========================
    with tab3:
        st.header("📈 Métricas y Evaluación")

        if st.session_state.get('entrenado'):
            modelo = st.session_state['modelo']
            X_test_scaled = st.session_state['X_test_scaled']
            y_test = st.session_state['y_test']

            y_pred = modelo.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Métricas en tarjetas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.1f}%")
            col2.metric("🔍 Precision", f"{precision:.4f}", f"{precision*100:.1f}%")
            col3.metric("📡 Recall", f"{recall:.4f}", f"{recall*100:.1f}%")
            col4.metric("⚖️ F1-Score", f"{f1:.4f}", f"{f1*100:.1f}%")

            st.subheader("Reporte de clasificación")
            report = classification_report(y_test, y_pred,
                target_names=['Desaprobado', 'Aprobado'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4),
                use_container_width=True)

            # Matriz de confusión
            st.subheader("Matriz de confusión")
            cm = confusion_matrix(y_test, y_pred)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Desaprobado', 'Aprobado'],
                yticklabels=['Desaprobado', 'Aprobado'],
                linewidths=1, linecolor='gray')
            axes[0].set_xlabel('Predicción')
            axes[0].set_ylabel('Valor Real')
            axes[0].set_title('Valores Absolutos')

            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Desaprobado', 'Aprobado'],
                yticklabels=['Desaprobado', 'Aprobado'],
                linewidths=1, linecolor='gray')
            axes[1].set_xlabel('Predicción')
            axes[1].set_ylabel('Valor Real')
            axes[1].set_title('Normalizada')
            plt.tight_layout()
            st.pyplot(fig)

            # Gráfica de métricas
            st.subheader("Comparativa visual de métricas")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            valores = [accuracy, precision, recall, f1]
            bars = ax3.bar(metricas, valores,
                color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
            ax3.set_ylim(0, 1.1)
            ax3.set_title(f'Métricas - {st.session_state["modelo_nombre"]}')
            for bar, val in zip(bars, valores):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.warning("⚠️ Primero entrena un modelo en la pestaña 'Entrenar Modelo'.")

    # ==========================
    # TAB 4: PREDICCIÓN
    # ==========================
    with tab4:
        st.header("🔮 Realizar Predicción con Nuevos Datos")

        if st.session_state.get('entrenado'):
            st.markdown("Ingresa los datos de un nuevo estudiante:")

            col1, col2 = st.columns(2)
            with col1:
                horas_est = st.number_input("📖 Horas de estudio semanal:", 0.0, 30.0, 15.0, 0.5)
                asistencia = st.number_input("📅 Asistencia (%):", 40.0, 100.0, 75.0, 1.0)
                promedio = st.number_input("📝 Promedio previo (0-20):", 0.0, 20.0, 12.0, 0.5)
                horas_sueno = st.number_input("😴 Horas de sueño diarias:", 3.0, 10.0, 7.0, 0.5)

            with col2:
                act_extra = st.selectbox("🏅 Actividades extracurriculares:", [0, 1],
                    format_func=lambda x: "Sí" if x == 1 else "No")
                nivel_socio = st.selectbox("💰 Nivel socioeconómico:", [1, 2, 3],
                    format_func=lambda x: {1: "Bajo", 2: "Medio", 3: "Alto"}[x])
                internet = st.selectbox("🌐 Acceso a internet:", [0, 1],
                    format_func=lambda x: "Sí" if x == 1 else "No")

            if st.button("🔮 PREDECIR", type="primary", use_container_width=True):
                nuevo = pd.DataFrame({
                    'horas_estudio': [horas_est],
                    'asistencia': [asistencia],
                    'promedio_previo': [promedio],
                    'horas_sueno': [horas_sueno],
                    'actividades_extra': [act_extra],
                    'nivel_socioeconomico': [nivel_socio],
                    'acceso_internet': [internet],
                })

                scaler = st.session_state['scaler']
                modelo = st.session_state['modelo']

                nuevo_scaled = scaler.transform(nuevo)
                pred = modelo.predict(nuevo_scaled)[0]
                prob = modelo.predict_proba(nuevo_scaled)[0]

                st.markdown("---")
                if pred == 1:
                    st.success(f"""
                    ### ✅ RESULTADO: APROBADO
                    - **Probabilidad de aprobar:** {prob[1]*100:.1f}%
                    - **Probabilidad de desaprobar:** {prob[0]*100:.1f}%
                    """)
                else:
                    st.error(f"""
                    ### ❌ RESULTADO: DESAPROBADO
                    - **Probabilidad de aprobar:** {prob[1]*100:.1f}%
                    - **Probabilidad de desaprobar:** {prob[0]*100:.1f}%
                    """)

                # Gauge visual
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh(['Probabilidad'], [prob[1]], color='#2ecc71' if pred == 1 else '#e74c3c', height=0.4)
                ax.barh(['Probabilidad'], [1-prob[1]], left=[prob[1]], color='#ecf0f1', height=0.4)
                ax.set_xlim(0, 1)
                ax.set_title(f'Probabilidad de Aprobación: {prob[1]*100:.1f}%')
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("⚠️ Primero entrena un modelo en la pestaña 'Entrenar Modelo'.")

    # ==========================
    # TAB 5: APRENDIZAJE
    # ==========================
    with tab5:
        st.header("📚 ¿Qué estás aprendiendo?")

        st.markdown("""
        ### 🧠 Conceptos clave de este laboratorio

        **1. Dataset y Variables**
        - Las **features** (características) son los datos de entrada del modelo
        - La **variable objetivo** es lo que queremos predecir (aprobado/desaprobado)

        **2. Preprocesamiento**
        - **StandardScaler** normaliza los datos para que todas las variables tengan la misma escala
        - La **división train/test** separa datos para entrenar y evaluar sin sesgo

        **3. Modelos disponibles**

        | Modelo | Cómo funciona | Cuándo usarlo |
        |--------|--------------|---------------|
        | **Logistic Regression** | Traza una línea que separa las clases | Datos linealmente separables |
        | **Decision Tree** | Crea reglas tipo "si X > 5 entonces..." | Datos con reglas claras |
        | **Random Forest** | Combina muchos árboles de decisión | Datos complejos, evita overfitting |
        | **SVM** | Busca el margen máximo entre clases | Datasets pequeños/medianos |
        | **KNN** | Clasifica según los K vecinos más cercanos | Datos con clusters definidos |

        **4. Métricas de evaluación**
        - **Accuracy:** ¿Qué porcentaje acerté en total?
        - **Precision:** De los que dije "aprobado", ¿cuántos realmente lo eran?
        - **Recall:** De todos los aprobados reales, ¿cuántos detecté?
        - **F1-Score:** Balance entre Precision y Recall

        **5. Experimenta**
        - 🎛️ Cambia los hiperparámetros y observa cómo cambian las métricas
        - 🔄 Prueba diferentes modelos con los mismos datos
        - 📊 Compara resultados para entender qué modelo funciona mejor
        """)

else:
    st.info("👈 Selecciona una fuente de datos en el panel lateral para comenzar.")