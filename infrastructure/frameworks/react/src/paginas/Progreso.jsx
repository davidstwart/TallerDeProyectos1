import Layout from "../components/layout/Layout";

import "../estilos/progress.css";

function Progreso() {

  // =====================================
  // MOCK DATA
  // =====================================

  const progressModules = [

    {
      titulo: "Machine Learning",
      avance: 72,
      estado: "Bueno",
    },

    {
      titulo: "Python IA",
      avance: 54,
      estado: "Regular",
    },

    {
      titulo: "Deep Learning",
      avance: 28,
      estado: "Riesgo",
    },
  ];

  return (

    <Layout>

      <div className="progress-page">

        {/* =================================
            HERO
        ================================== */}

        <section className="progress-hero">

          <div>

            <span className="progress-badge">
              Seguimiento IA
            </span>

            <h1>
              Progreso académico
            </h1>

            <p>
              Visualiza rendimiento,
              avance de cursos,
              métricas educativas y
              predicciones académicas.
            </p>

          </div>

        </section>

        {/* =================================
            METRICS
        ================================== */}

        <section className="progress-metrics">

          <article className="metric-card">

            <strong>
              78%
            </strong>

            <span>
              Avance general
            </span>

          </article>

          <article className="metric-card">

            <strong>
              16.8
            </strong>

            <span>
              Promedio
            </span>

          </article>

          <article className="metric-card">

            <strong>
              Bajo
            </strong>

            <span>
              Riesgo académico
            </span>

          </article>

          <article className="metric-card">

            <strong>
              IA
            </strong>

            <span>
              Predicción activa
            </span>

          </article>

        </section>

        {/* =================================
            MODULES
        ================================== */}

        <section className="progress-section">

          <div className="section-header">

            <h2>
              Avance por curso
            </h2>

            <p>
              Seguimiento del
              aprendizaje por tema.
            </p>

          </div>

          <div className="progress-list">

            {
              progressModules.map(
                (module, index) => (

                  <article
                    key={index}
                    className="progress-card"
                  >

                    <div className="progress-top">

                      <div className="progress-icon">

                        {
                          module.titulo
                            .charAt(0)
                        }

                      </div>

                      <span
                        className={

                          module.estado ===
                          "Bueno"

                            ? "status success"

                            : module.estado ===
                              "Regular"

                            ? "status warning"

                            : "status danger"
                        }
                      >

                        {module.estado}

                      </span>

                    </div>

                    <h3>
                      {module.titulo}
                    </h3>

                    <div className="course-progress">

                      <div className="progress-info">

                        <span>
                          Progreso
                        </span>

                        <strong>
                          {
                            module.avance
                          }%
                        </strong>

                      </div>

                      <div className="progress-bar">

                        <div
                          style={{
                            width:
                              `${module.avance}%`,
                          }}
                        />

                      </div>

                    </div>

                  </article>
                )
              )
            }

          </div>

        </section>

        {/* =================================
            IA RECOMMENDATIONS
        ================================== */}

        <section className="recommendation-section">

          <div className="section-header">

            <h2>
              Recomendaciones IA
            </h2>

            <p>
              Sugerencias automáticas
              basadas en rendimiento.
            </p>

          </div>

          <div className="recommendation-grid">

            <article className="recommendation-card">

              <span>
                🤖
              </span>

              <h3>
                Mejorar Deep Learning
              </h3>

              <p>
                El sistema detectó bajo
                avance en redes neuronales.
                Se recomienda reforzar
                ejercicios prácticos.
              </p>

            </article>

            <article className="recommendation-card">

              <span>
                📚
              </span>

              <h3>
                Reforzar datasets
              </h3>

              <p>
                Incrementa prácticas
                relacionadas al análisis
                y limpieza de datos.
              </p>

            </article>

            <article className="recommendation-card">

              <span>
                🧠
              </span>

              <h3>
                Continuar entrenamiento
              </h3>

              <p>
                Tu rendimiento en Machine
                Learning es positivo.
                Continúa avanzando.
              </p>

            </article>

          </div>

        </section>

      </div>

    </Layout>
  );
}

export default Progreso;