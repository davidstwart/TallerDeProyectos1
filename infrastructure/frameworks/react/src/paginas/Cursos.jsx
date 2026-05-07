import {
  useEffect,
  useState,
} from "react";

import Layout from "../components/layout/Layout";

import {
  getTemasRequest,
} from "../services/temaService";

import "../estilos/courses.css";

function Cursos() {

  // =====================================
  // STATES
  // =====================================

  const [temas, setTemas] =
    useState([]);

  const [loading, setLoading] =
    useState(true);

  // =====================================
  // LOAD TEMAS
  // =====================================

  const loadTemas = async () => {

    try {

      const data =
        await getTemasRequest();

      setTemas(data);

    } catch (error) {

      console.error(error);

    } finally {

      setLoading(false);
    }

  };

  useEffect(() => {

    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadTemas();

  }, []);

  return (

    <Layout>

      <div className="courses-page">

        {/* HERO */}

        <section className="courses-hero">

          <div>

            <span className="courses-badge">
              Curso oficial
            </span>

            <h1>
              Computación e Informática
            </h1>

            <p>
              Gestión de temas educativos,
              inteligencia artificial,
              datasets y aprendizaje
              automático.
            </p>

          </div>

        </section>

        {/* TEMAS */}

        <section className="courses-grid">

          {
            loading
              ? (
                <p>
                  Cargando temas...
                </p>
              )
              : temas.map((tema) => (

                <article
                  key={tema.id_tema}
                  className="course-card"
                >

                  <div className="course-top">

                    <div className="course-icon blue">

                      {
                        tema.nombre.charAt(0)
                      }

                    </div>

                    <span className="course-tag">

                      Tema

                    </span>

                  </div>

                  <h3>
                    {tema.nombre}
                  </h3>

                  <p>
                    {tema.descripcion}
                  </p>

                  <button className="course-btn">

                    Ver tema

                  </button>

                </article>
              ))
          }

        </section>

      </div>

    </Layout>
  );
}

export default Cursos;