import {
  useEffect,
  useState,
} from "react";

import Layout from "../components/layout/Layout";

import {
  createUsuarioRequest,
  getDocentesRequest,
} from "../services/usuarioService";

import "../estilos/students.css";

function AdminDocentes() {

  // =====================================
  // STATES
  // =====================================

  const [docentes, setDocentes] =
    useState([]);

  const [loading, setLoading] =
    useState(true);

  const [creating, setCreating] =
    useState(false);

  const [error, setError] =
    useState("");

  const [success, setSuccess] =
    useState("");

  const [formData, setFormData] =
    useState({

      nombres: "",

      apellidos: "",

      fecha_nacimiento: "",

      correo: "",

      celular: "",

      password: "",

      grado: "",

      seccion: "",

      id_rol: 2,
    });

  // =====================================
  // LOAD DOCENTES
  // =====================================

  const loadDocentes =
    async () => {

      try {

        const data =
          await getDocentesRequest();

        setDocentes(data);

      } catch (error) {

        console.error(error);

      } finally {

        setLoading(false);
      }
    };

  useEffect(() => {

    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadDocentes();

  }, []);

  // =====================================
  // INPUT CHANGE
  // =====================================

  const handleChange = (
    e
  ) => {

    setFormData({

      ...formData,

      [e.target.name]:
        e.target.value,
    });
  };

  // =====================================
  // CREATE DOCENTE
  // =====================================

  const handleCreateDocente =
    async (e) => {

      e.preventDefault();

      try {

        setCreating(true);

        setError("");

        setSuccess("");

        await createUsuarioRequest(
          formData
        );

        setSuccess(
          "Docente creado correctamente"
        );

        setFormData({

          nombres: "",

          apellidos: "",

          fecha_nacimiento: "",

          correo: "",

          celular: "",

          password: "",

          grado: "",

          seccion: "",

          id_rol: 2,
        });

        loadDocentes();

      } catch (error) {

        console.error(error);

        setError(
          error.response?.data?.detail ||
          "Error creando docente"
        );

      } finally {

        setCreating(false);
      }
    };

  return (

    <Layout>

      <div className="students-page">

        {/* HEADER */}

        <section className="students-header">

          <div>

            <span className="students-badge">
              Administración
            </span>

            <h1>
              Gestión de docentes
            </h1>

            <p>
              Crea y administra
              docentes registrados
              en la plataforma.
            </p>

          </div>

        </section>

        {/* ALERTS */}

        {
          error && (

            <div className="lab-card">

              <p
                style={{
                  color: "#fca5a5",
                }}
              >
                {error}
              </p>

            </div>
          )
        }

        {
          success && (

            <div className="lab-card">

              <p
                style={{
                  color: "#4ade80",
                }}
              >
                {success}
              </p>

            </div>
          )
        }

        {/* FORM */}

        <section className="lab-card">

          <div className="card-title">

            <span>01</span>

            <div>

              <h2>
                Crear docente
              </h2>

              <p>
                Registrar nuevo docente.
              </p>

            </div>

          </div>

          <form
            onSubmit={
              handleCreateDocente
            }
            className="form-grid"
          >

            <div className="lab-form-group">

              <label>
                Nombres
              </label>

              <input
                type="text"
                name="nombres"
                value={formData.nombres}
                onChange={handleChange}
                required
              />

            </div>

            <div className="lab-form-group">

              <label>
                Apellidos
              </label>

              <input
                type="text"
                name="apellidos"
                value={formData.apellidos}
                onChange={handleChange}
                required
              />

            </div>

            <div className="lab-form-group">

              <label>
                Fecha nacimiento
              </label>

              <input
                type="date"
                name="fecha_nacimiento"
                value={formData.fecha_nacimiento}
                onChange={handleChange}
                required
              />

            </div>

            <div className="lab-form-group">

              <label>
                Correo
              </label>

              <input
                type="email"
                name="correo"
                value={formData.correo}
                onChange={handleChange}
                required
              />

            </div>

            <div className="lab-form-group">

              <label>
                Celular
              </label>

              <input
                type="text"
                name="celular"
                value={formData.celular}
                onChange={handleChange}
                required
              />

            </div>

            <div className="lab-form-group">

              <label>
                Contraseña
              </label>

              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
              />

            </div>

            <button
              type="submit"
              className="train-btn"
              disabled={creating}
              style={{
                gridColumn:
                  "1 / -1",
              }}
            >

              {
                creating
                  ? "Creando..."
                  : "Crear docente"
              }

            </button>

          </form>

        </section>

        {/* DOCENTES */}

        <section className="students-grid">

          {
            loading
              ? (
                <p>
                  Cargando docentes...
                </p>
              )
              : docentes.map(
                (docente) => (

                  <article
                    key={docente.id_usuario}
                    className="student-card"
                  >

                    <div className="student-top">

                      <div className="student-avatar">

                        {
                          docente.nombres
                            .charAt(0)
                        }

                      </div>

                      <span className="status active">

                        Docente

                      </span>

                    </div>

                    <h3>

                      {
                        docente.nombres
                      } {

                        docente.apellidos
                      }

                    </h3>

                    <p>
                      {
                        docente.correo
                      }
                    </p>

                    <div className="student-metrics">

                      <div>

                        <strong>
                          {
                            docente.celular
                          }
                        </strong>

                        <span>
                          Celular
                        </span>

                      </div>

                    </div>

                  </article>
                )
              )
          }

        </section>

      </div>

    </Layout>
  );
}

export default AdminDocentes;