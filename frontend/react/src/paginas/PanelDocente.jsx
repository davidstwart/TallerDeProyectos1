import {
  Link,
} from "react-router-dom";

import Layout from "../components/layout/Layout";

import "../estilos/dashboard.css";

function PanelDocente() {

  return (

    <Layout>

      <div className="dashboard-page">

        {/* =====================================
            HERO
        ====================================== */}

        <section className="dashboard-hero">

          <div>

            <span className="dashboard-badge">
              Panel Docente
            </span>

            <h1>
              Gestiona el aprendizaje
              y entrenamiento IA
            </h1>

            <p>
              Administra estudiantes,
              cursos, datasets y
              modelos de inteligencia
              artificial para potenciar
              el aprendizaje académico.
            </p>

          </div>

        </section>

        {/* =====================================
            STATS
        ====================================== */}

        <section className="dashboard-stats">

          <article className="stat-card">

            <strong>
              124
            </strong>

            <span>
              Estudiantes
            </span>

          </article>

          <article className="stat-card">

            <strong>
              12
            </strong>

            <span>
              Cursos activos
            </span>

          </article>

          <article className="stat-card">

            <strong>
              8
            </strong>

            <span>
              Modelos entrenados
            </span>

          </article>

          <article className="stat-card">

            <strong>
              92%
            </strong>

            <span>
              Accuracy promedio
            </span>

          </article>

        </section>

        {/* =====================================
            MODULES
        ====================================== */}

        <section className="dashboard-section">

          <div className="section-header">

            <h2>
              Módulos principales
            </h2>

            <p>
              Accede rápidamente
              a las herramientas
              educativas.
            </p>

          </div>

          <div className="dashboard-grid">

            {/* LAB */}

            <Link
              to="/laboratorio"
              className="dashboard-card"
            >

              <div className="dashboard-icon">
                🧠
              </div>

              <h3>
                Laboratorio IA
              </h3>

              <p>
                Entrena modelos,
                sube datasets y
                genera predicciones.
              </p>

            </Link>

            {/* CURSOS */}

            <Link
              to="/cursos"
              className="dashboard-card"
            >

              <div className="dashboard-icon">
                📚
              </div>

              <h3>
                Cursos
              </h3>

              <p>
                Gestiona cursos,
                módulos y contenidos
                educativos.
              </p>

            </Link>

            {/* ESTUDIANTES */}

            <Link
              to="/estudiantes"
              className="dashboard-card"
            >

              <div className="dashboard-icon">
                👨‍🎓
              </div>

              <h3>
                Estudiantes
              </h3>

              <p>
                Visualiza rendimiento,
                progreso y métricas
                académicas.
              </p>

            </Link>

            {/* MODELOS */}

            <Link
              to="/modelos"
              className="dashboard-card"
            >

              <div className="dashboard-icon">
                🤖
              </div>

              <h3>
                Modelos
              </h3>

              <p>
                Consulta modelos
                entrenados y sus
                métricas.
              </p>

            </Link>

          </div>

        </section>

      </div>

    </Layout>
  );
}

export default PanelDocente;