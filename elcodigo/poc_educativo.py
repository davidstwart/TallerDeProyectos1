import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

np.random.seed(42)

# --- GENERACIÓN DEL DATASET ---
n_estudiantes = 500

datos = {
    'horas_estudio': np.random.uniform(0, 30, n_estudiantes),
    'asistencia': np.random.uniform(40, 100, n_estudiantes),
    'promedio_previo': np.random.uniform(5, 20, n_estudiantes),
    'horas_sueno': np.random.uniform(3, 10, n_estudiantes),
    'actividades_extra': np.random.choice([0, 1], n_estudiantes, p=[0.4, 0.6]),
    'nivel_socioeconomico': np.random.choice([1, 2, 3], n_estudiantes, p=[0.3, 0.5, 0.2]),
    'acceso_internet': np.random.choice([0, 1], n_estudiantes, p=[0.2, 0.8]),
}

df = pd.DataFrame(datos)

# --- VARIABLE OBJETIVO ---
puntaje = (
    0.25 * df['horas_estudio'] / 30 +
    0.20 * df['asistencia'] / 100 +
    0.25 * df['promedio_previo'] / 20 +
    0.10 * df['horas_sueno'] / 10 +
    0.08 * df['actividades_extra'] +
    0.07 * df['nivel_socioeconomico'] / 3 +
    0.05 * df['acceso_internet']
)
ruido = np.random.normal(0, 0.05, n_estudiantes)
puntaje += ruido
df['aprobado'] = (puntaje >= 0.55).astype(int)

# --- EDA ---
print("=" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 60)
print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nPrimeras 5 filas:")
print(df.head())
print(f"\nEstadísticas descriptivas:")
print(df.describe().round(2))
print(f"\nDistribución de la variable objetivo:")
print(df['aprobado'].value_counts())
print(f"\nPorcentaje de aprobados: {df['aprobado'].mean()*100:.1f}%")
print(f"Porcentaje de desaprobados: {(1-df['aprobado'].mean())*100:.1f}%")
print(f"\nValores nulos por columna:")
print(df.isnull().sum())

# --- PREPROCESAMIENTO ---
X = df.drop('aprobado', axis=1)
y = df['aprobado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n{'='*60}")
print("DIVISIÓN DE DATOS")
print(f"{'='*60}")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba:        {X_test.shape[0]} muestras")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- ENTRENAMIENTO ---
print(f"\n{'='*60}")
print("ENTRENAMIENTO DEL MODELO")
print(f"{'='*60}")

modelo = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=1.0)
modelo.fit(X_train_scaled, y_train)
print("Modelo entrenado exitosamente: Logistic Regression")

coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo.coef_[0].round(4)
}).sort_values('Coeficiente', ascending=False)

print(f"\nCoeficientes del modelo:")
print(coeficientes.to_string(index=False))
print(f"\nIntercepto: {modelo.intercept_[0]:.4f}")

# --- EVALUACIÓN ---
y_pred = modelo.predict(X_test_scaled)
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n{'='*60}")
print("MÉTRICAS DE EVALUACIÓN")
print(f"{'='*60}")
print(f"  Accuracy  (Exactitud):    {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (Precisión):    {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    (Sensibilidad): {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score  (Media armón.): {f1:.4f}  ({f1*100:.2f}%)")

print(f"\nReporte de clasificación completo:")
print(classification_report(y_test, y_pred, target_names=['Desaprobado', 'Aprobado']))

# --- MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Desaprobado', 'Aprobado'],
            yticklabels=['Desaprobado', 'Aprobado'],
            linewidths=1, linecolor='gray')
axes[0].set_xlabel('Predicción', fontsize=12)
axes[0].set_ylabel('Valor Real', fontsize=12)
axes[0].set_title('Matriz de Confusión (Valores Absolutos)', fontsize=13, fontweight='bold')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=['Desaprobado', 'Aprobado'],
            yticklabels=['Desaprobado', 'Aprobado'],
            linewidths=1, linecolor='gray')
axes[1].set_xlabel('Predicción', fontsize=12)
axes[1].set_ylabel('Valor Real', fontsize=12)
axes[1].set_title('Matriz de Confusión (Normalizada)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# --- VISUALIZACIONES ADICIONALES ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#e74c3c', '#2ecc71']
df['aprobado'].value_counts().plot(kind='bar', ax=axes[0, 0], color=colors)
axes[0, 0].set_title('Distribución: Aprobados vs Desaprobados', fontweight='bold')
axes[0, 0].set_xticklabels(['Desaprobado', 'Aprobado'], rotation=0)
axes[0, 0].set_ylabel('Cantidad')

coef_sorted = coeficientes.sort_values('Coeficiente')
axes[0, 1].barh(coef_sorted['Variable'], coef_sorted['Coeficiente'],
                color=['#e74c3c' if c < 0 else '#2ecc71' for c in coef_sorted['Coeficiente']])
axes[0, 1].set_title('Importancia de Variables (Coeficientes)', fontweight='bold')
axes[0, 1].axvline(x=0, color='black', linewidth=0.8)

axes[1, 0].scatter(df[df['aprobado']==0]['horas_estudio'],
                   df[df['aprobado']==0]['promedio_previo'],
                   c='#e74c3c', alpha=0.5, label='Desaprobado', s=20)
axes[1, 0].scatter(df[df['aprobado']==1]['horas_estudio'],
                   df[df['aprobado']==1]['promedio_previo'],
                   c='#2ecc71', alpha=0.5, label='Aprobado', s=20)
axes[1, 0].set_xlabel('Horas de Estudio')
axes[1, 0].set_ylabel('Promedio Previo')
axes[1, 0].set_title('Horas de Estudio vs Promedio Previo', fontweight='bold')
axes[1, 0].legend()

metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
valores = [accuracy, precision, recall, f1]
bars = axes[1, 1].bar(metricas, valores, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_title('Métricas de Evaluación del Modelo', fontweight='bold')
for bar, val in zip(bars, valores):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_completo.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PREDICCIÓN CON NUEVOS DATOS ---
print(f"\n{'='*60}")
print("PREDICCIÓN CON NUEVOS ESTUDIANTES")
print(f"{'='*60}")

nuevos_estudiantes = pd.DataFrame({
    'horas_estudio':         [25, 5, 15],
    'asistencia':            [95, 50, 75],
    'promedio_previo':       [17, 8, 12],
    'horas_sueno':           [8, 4, 6],
    'actividades_extra':     [1, 0, 1],
    'nivel_socioeconomico':  [3, 1, 2],
    'acceso_internet':       [1, 0, 1],
}, index=['Estudiante A', 'Estudiante B', 'Estudiante C'])

print("\nDatos de los nuevos estudiantes:")
print(nuevos_estudiantes)

nuevos_scaled = scaler.transform(nuevos_estudiantes)
predicciones = modelo.predict(nuevos_scaled)
probabilidades = modelo.predict_proba(nuevos_scaled)[:, 1]

print(f"\nResultados de la predicción:")
print("-" * 50)
for i, nombre in enumerate(nuevos_estudiantes.index):
    estado = "APROBADO ✓" if predicciones[i] == 1 else "DESAPROBADO ✗"
    print(f"  {nombre}: {estado} (Probabilidad de aprobar: {probabilidades[i]*100:.1f}%)")

print(f"\n{'='*60}")
print("FIN DEL POC")
print(f"{'='*60}")