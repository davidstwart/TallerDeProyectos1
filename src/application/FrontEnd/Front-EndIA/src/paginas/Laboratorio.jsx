import "../estilos/pages.css";

function Laboratorio() {
  return (
    <div className="lab-page">
      <header className="lab-header">
        <div>
          <span className="lab-badge">Laboratorio IA</span>
          <h1>Entrena un modelo con tu propio dataset</h1>
          <p>
            Sube un archivo de datos, selecciona una variable objetivo y simula
            el entrenamiento de un modelo de inteligencia artificial.
          </p>
        </div>
      </header>

      <main className="lab-layout">
        <section className="lab-card upload-card">
          <div className="card-title">
            <span>01</span>
            <div>
              <h2>Subir dataset</h2>
              <p>Formatos permitidos: CSV o Excel.</p>
            </div>
          </div>

          <label className="upload-box">
            <input type="file" accept=".csv,.xlsx" />
            <div className="upload-icon">📁</div>
            <h3>Selecciona o arrastra tu archivo</h3>
            <p>Ejemplo: estudiantes.csv, ventas.xlsx, datos_ia.csv</p>
          </label>
        </section>

        <section className="lab-card config-card">
          <div className="card-title">
            <span>02</span>
            <div>
              <h2>Configurar entrenamiento</h2>
              <p>Elige cómo se entrenará el modelo.</p>
            </div>
          </div>

          <div className="form-grid">
            <div className="lab-form-group">
              <label>Variable objetivo</label>
              <select>
                <option>Seleccionar columna</option>
                <option>Aprobado</option>
                <option>Nota final</option>
                <option>Categoría</option>
              </select>
            </div>

            <div className="lab-form-group">
              <label>Tipo de problema</label>
              <select>
                <option>Clasificación</option>
                <option>Predicción numérica</option>
              </select>
            </div>

            <div className="lab-form-group">
              <label>Algoritmo</label>
              <select>
                <option>Árbol de decisión</option>
                <option>Random Forest</option>
                <option>Regresión logística</option>
              </select>
            </div>

            <div className="lab-form-group">
              <label>División de datos</label>
              <select>
                <option>80% entrenamiento / 20% prueba</option>
                <option>70% entrenamiento / 30% prueba</option>
                <option>60% entrenamiento / 40% prueba</option>
              </select>
            </div>
          </div>

          <button className="train-btn">Iniciar entrenamiento</button>
        </section>

        <section className="lab-card preview-card">
          <div className="card-title">
            <span>03</span>
            <div>
              <h2>Vista previa del dataset</h2>
              <p>Primeras filas del archivo cargado.</p>
            </div>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Edad</th>
                  <th>Horas estudio</th>
                  <th>Nota</th>
                  <th>Aprobado</th>
                </tr>
              </thead>

              <tbody>
                <tr>
                  <td>15</td>
                  <td>4</td>
                  <td>16</td>
                  <td>Sí</td>
                </tr>
                <tr>
                  <td>16</td>
                  <td>2</td>
                  <td>12</td>
                  <td>Sí</td>
                </tr>
                <tr>
                  <td>15</td>
                  <td>1</td>
                  <td>09</td>
                  <td>No</td>
                </tr>
                <tr>
                  <td>17</td>
                  <td>5</td>
                  <td>18</td>
                  <td>Sí</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="lab-card results-card">
          <div className="card-title">
            <span>04</span>
            <div>
              <h2>Resultados del modelo</h2>
              <p>Resultados simulados del entrenamiento.</p>
            </div>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <strong>89%</strong>
              <span>Accuracy</span>
            </div>

            <div className="metric">
              <strong>85%</strong>
              <span>Precisión</span>
            </div>

            <div className="metric">
              <strong>87%</strong>
              <span>Recall</span>
            </div>
          </div>

          <div className="result-message">
            <span>✅</span>
            <p>
              Modelo entrenado correctamente. El sistema identificó patrones en
              el dataset cargado.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Laboratorio;