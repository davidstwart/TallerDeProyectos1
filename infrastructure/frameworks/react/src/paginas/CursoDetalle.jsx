import {
  useParams,
  Link,
} from "react-router-dom";

import Layout from "../components/layout/Layout";

import "../estilos/course-detail.css";

function CursoDetalle() {

  const {
    id,
  } = useParams();

  // =====================================
  // MOCK DATA
  // =====================================

  const modules = [

    {
      id: 1,
      titulo: "Introducción al Machine Learning",
      descripcion:
        "Conceptos fundamentales del aprendizaje automático.",

      progreso: 100,

      estado: "Completado",
    },

    {
      id: 2,
      titulo: "Preparación de datasets",
      descripcion:
        "Limpieza, transformación y análisis de datos.",

      progreso: 82,

      estado: "En progreso",
    },

    {
      id: 3,
      titulo: "Entrenamiento de modelos",
      descripcion:
        "Uso de algoritmos supervisados y métricas.",

      progreso: 45,

      estado: "En progreso",
    },

    {
      id: 4,
      titulo: "Predicciones y evaluación",
      descripcion:
        "Evaluación de accuracy, precision y recall.",

      progreso: 0,

      estado: "Pendiente",
    },
  ];

  return (

    <Layout>

      <div className="course-detail-page">

        {/* =================================
            HERO
        ================================== */}

        <section className="course-detail-hero">

          <div>

            <span className="detail-badge">

              Curso IA #{id}

            </span>

            <h1>
              Machine Learning
            </h1>

            <p>
              Aprende modelos predictivos,
              datasets, clasificación,
              entrenamiento y evaluación
              de algoritmos de IA.
            </p>

          </div>

          <div className="hero-actions">

            <Link
              to="/laboratorio"
              className="primary-btn"
            >

              Ir al laboratorio

            </Link>

          </div>

        </section>

        {/* =================================
            STATS
        ================================== */}

        <section className="detail-stats">

          <article>

            <strong>
              72%
            </strong>

            <span>
              Progreso
            </span>

          </article>

          <article>

            <strong>
              12
            </strong>

            <span>
              Módulos
            </span>

          </article>

          <article>

            <strong>
              48
            </strong>

            <span>
              Estudiantes
            </span>

          </article>

          <article>

            <strong>
              8
            </strong>

            <span>
              Actividades
            </span>

          </article>

        </section>

        {/* =================================
            MODULES
        ================================== */}

        <section className="modules-section">

          <div className="section-header">

            <h2>
              Módulos del curso
            </h2>

            <p>
              Sigue el progreso de
              aprendizaje y actividades.
            </p>

          </div>

          <div className="modules-list">

            {
              modules.map((module) => (

                <article
                  key={module.id}
                  className="module-card"
                >

                  <div className="module-top">

                    <div className="module-number">

                      {module.id}

                    </div>

                    <span
                      className={

                        module.estado ===
                        "Completado"

                          ? "module-status completed"

                          : module.estado ===
                            "En progreso"

                          ? "module-status progress"

                          : "module-status pending"
                      }
                    >

                      {module.estado}

                    </span>

                  </div>

                  <h3>
                    {module.titulo}
                  </h3>

                  <p>
                    {
                      module.descripcion
                    }
                  </p>

                  {/* PROGRESS */}

                  <div className="module-progress">

                    <div className="progress-info">

                      <span>
                        Avance
                      </span>

                      <strong>
                        {
                          module.progreso
                        }%
                      </strong>

                    </div>

                    <div className="progress-bar">

                      <div
                        style={{
                          width:
                            `${module.progreso}%`,
                        }}
                      />

                    </div>

                  </div>

                  <button className="module-btn">

                    Ver módulo

                  </button>

                </article>
              ))
            }

          </div>

        </section>

      </div>

    </Layout>
  );
}

export default CursoDetalle;