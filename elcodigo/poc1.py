# ================================
# POC IA EDUCATIVA - PMV 1
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. CREAR DATASET SIMULADO
# -----------------------------
np.random.seed(500)

data = {
    'horas_estudio': np.random.randint(1, 10, 100),
    'asistencia': np.random.randint(50, 100, 100),
    'tareas_completadas': np.random.randint(0, 10, 100),
    'participacion': np.random.randint(0, 10, 100),
}

df = pd.DataFrame(data)

# Regla simulada (target)
df['resultado_aprobado'] = (
    (df['horas_estudio'] > 4) &
    (df['asistencia'] > 70) &
    (df['tareas_completadas'] > 5)
).astype(int)

print("Dataset generado:")
print(df.head())

# -----------------------------
# 2. PREPARAR DATOS
# -----------------------------
X = df.drop('resultado_aprobado', axis=1)
y = df['resultado_aprobado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# 3. ENTRENAMIENTO
# -----------------------------
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

print("\nModelo entrenado correctamente")

# -----------------------------
# 4. PREDICCIÓN
# -----------------------------
y_pred = modelo.predict(X_test)

# -----------------------------
# 5. EVALUACIÓN
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\n=== MÉTRICAS ===")
print(f"Exactitud (Accuracy): {accuracy}")
print(f"Precisión (Precision): {precision}")

print("\nReporte completo:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. MATRIZ DE CONFUSIÓN
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# -----------------------------
# 7. PREDICIÓN NUEVA
# -----------------------------
nuevo_estudiante = [[6, 80, 7, 5]]  # ejemplo
pred = modelo.predict(nuevo_estudiante)

print("\nPredicción para nuevo estudiante:")
print("Aprobado" if pred[0] == 1 else "Desaprobado")