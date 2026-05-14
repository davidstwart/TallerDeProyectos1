import {
  useParams,
  Link,
} from "react-router-dom";

import {
  useEffect,
  useState,
} from "react";

import Layout from "../components/layout/Layout";

import {
  getUsuariosRequest,
} from "../services/usuarioService";

import "../estilos/student-detail.css";

function EstudianteDetalle() {

  const {
    id,
  } = useParams();

  // =====================================
  // STATES
  // =====================================

  const [student, setStudent] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  // =====================================
  // LOAD STUDENT
  // =====================================

  useEffect(() => {

    async function loadStudent() {

      try {

        const users =
          await getUsuariosRequest();

        const foundStudent =
          users.find(

            (u) =>

              u.id_usuario ===
              Number(id)

          );

        setStudent(
          foundStudent
        );

      } catch (error) {

        console.error(error);

      } finally {

        setLoading(false);
      }
    }

    loadStudent();

  }, [id]);

  // =====================================
  // LOADING
  // =====================================

  if (loading) {

    return (

      <Layout>

        <p>
          Cargando estudiante...
        </p>

      </Layout>
    );
  }

  // =====================================
  // NOT FOUND
  // =====================================

  if (!student) {

    return (

      <Layout>

        <p>
          Estudiante no encontrado
        </p>

      </Layout>
    );
  }

  // =====================================
  // MOCK IA DATA
  // =====================================

  const metrics = {

    promedio: 16.8,

    asistencia: 92,

    progreso: 78,

    riesgo:
      "Bajo",

    probabilidad:
      87,

  };

  const temas = [

    {
      nombre:
        "Machine Learning",

      progreso: 82,
    },

    {
      nombre:
        "Python para IA",

      progreso: 67,
    },

    {
      nombre:
        "Deep Learning",

      progreso: 41,
    },
  ];

  const recomendaciones = [

    "Continuar reforzando Machine Learning.",

    "Mejorar práctica semanal en Python.",

    "Mantener asistencia superior al 90%.",

  ];

  return (

    <Layout>

      <div className="student-detail-page">

        {/* =================================
            HERO
        ================================== */}

        <section className="student-hero">

          <div>

            <Link
              to="/estudiantes"
              className="back-link"
            >

              ← Volver

            </Link>

            <span className="student-badge">

              Perfil académico

            </span>

            <h1>

              {
                student.nombres
              } {

                student.apellidos
              }

            </h1>

            <p>

              {
                student.grado
              }

              {" - Sección "}

              {
                student.seccion
              }

            </p>

          </div>

          <div className="student-risk">

            <strong>

              {
                metrics.riesgo
              }

            </strong>

            <span>
              Riesgo IA
            </span>

          </div>

        </section>

        {/* =================================
            METRICS
        ================================== */}

        <section className="student-metrics-grid">

          <article className="metric-card">

            <strong>
              {
                metrics.promedio
              }
            </strong>

            <span>
              Promedio
            </span>

          </article>

          <article className="metric-card">

            <strong>
              {
                metrics.asistencia
              }%
            </strong>

            <span>
              Asistencia
            </span>

          </article>

          <article className="metric-card">

            <strong>
              {
                metrics.progreso
              }%
            </strong>

            <span>
              Progreso
            </span>

          </article>

          <article className="metric-card">

            <strong>
              {
                metrics.probabilidad
              }%
            </strong>

            <span>
              Probabilidad de aprobar
            </span>

          </article>

        </section>

        {/* =================================
            TEMAS
        ================================== */}

        <section className="student-topics">

          <div className="section-title">

            <h2>
              Progreso por temas
            </h2>

            <p>
              Seguimiento académico
              del estudiante.
            </p>

          </div>

          <div className="topics-list">

            {
              temas.map((tema) => (

                <article
                  key={tema.nombre}
                  className="topic-card"
                >

                  <div>

                    <h3>
                      {
                        tema.nombre
                      }
                    </h3>

                    <p>
                      Avance del tema
                    </p>

                  </div>

                  <div className="topic-progress">

                    <strong>
                      {
                        tema.progreso
                      }%
                    </strong>

                    <div className="progress-bar">

                      <div
                        style={{
                          width:
                            `${tema.progreso}%`,
                        }}
                      />

                    </div>

                  </div>

                </article>
              ))
            }

          </div>

        </section>

        {/* =================================
            IA RECOMMENDATIONS
        ================================== */}

        <section className="recommendations-section">

          <div className="section-title">

            <h2>
              Recomendaciones IA
            </h2>

            <p>
              Sugerencias generadas
              mediante análisis académico.
            </p>

          </div>

          <div className="recommendations-list">

            {
              recomendaciones.map(
                (
                  recommendation,
                  index
                ) => (

                  <article
                    key={index}
                    className="recommendation-card"
                  >

                    <span>
                      🤖
                    </span>

                    <p>
                      {
                        recommendation
                      }
                    </p>

                  </article>
                )
              )
            }

          </div>

        </section>

      </div>

    </Layout>
  );
}

export default EstudianteDetalle;