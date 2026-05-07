import {
  Link,
} from "react-router-dom";
import {
  useEffect,
  useState,
} from "react";

import Layout from "../components/layout/Layout";

import {

  getEstudiantesRequest,
  createUsuarioRequest,

} from "../services/usuarioService";

import "../estilos/students.css";

function GestionEstudiantes() {

  // =====================================
  // STATES
  // =====================================

  const [students, setStudents] =
    useState([]);

  const [loading, setLoading] =
    useState(true);

  const [search, setSearch] =
    useState("");

  const [showModal, setShowModal] =
    useState(false);

  const [saving, setSaving] =
    useState(false);

  const [formData, setFormData] =
    useState({

      nombres: "",

      apellidos: "",

      fecha_nacimiento: "",

      grado: "",

      seccion: "",

      correo: "",

      celular: "",

      password: "",

      id_rol: 3,

    });

  // =====================================
  // LOAD STUDENTS
  // =====================================

  async function loadStudents() {

    try {

      const data =
        await getEstudiantesRequest();

      setStudents(data);

    } catch (error) {

      console.error(error);

    } finally {

      setLoading(false);
    }
  }

  // =====================================
  // HANDLE CHANGE
  // =====================================

  const handleChange = (e) => {

    setFormData({

      ...formData,

      [e.target.name]:
        e.target.value,

    });
  };

  async function createStudent() {

    try {

      setSaving(true);

      await createUsuarioRequest(
        formData
      );

      await loadStudents();

      setShowModal(false);

      setFormData({

        nombres: "",

        apellidos: "",

        fecha_nacimiento: "",

        grado: "",

        seccion: "",

        correo: "",

        celular: "",

        password: "",

        id_rol: 3,
      });

    } catch (error) {

      console.error(error);

      alert(
        error.response?.data?.detail ||
        "Error creando estudiante"
      );

    } finally {

      setSaving(false);
    }
  }

  // =====================================
  // CREATE STUDENT
  // =====================================

 

  // =====================================
  // EFFECT
  // =====================================

  useEffect(() => {

    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadStudents();

  }, []);

  // =====================================
  // FILTER
  // =====================================

  const filteredStudents =
    students.filter((student) =>

      `${student.nombres} ${student.apellidos}`
        .toLowerCase()
        .includes(
          search.toLowerCase()
        )
    );

  return (

    <Layout>

      <div className="students-page">

        {/* =================================
            HEADER
        ================================== */}

        <section className="students-header">

          <div>

            <span className="students-badge">
              Gestión Académica
            </span>

            <h1>
              Estudiantes registrados
            </h1>

            <p>
              Visualiza estudiantes,
              seguimiento académico y
              progreso educativo.
            </p>

          </div>

          <button
            className="primary-btn"
            onClick={() =>
              setShowModal(true)
            }
          >

            + Nuevo estudiante

          </button>

        </section>

        {/* =================================
            SEARCH
        ================================== */}

        <section className="students-toolbar">

          <input
            type="text"
            placeholder="Buscar estudiante..."
            value={search}
            onChange={(e) =>
              setSearch(
                e.target.value
              )
            }
          />

        </section>

        {/* =================================
            GRID
        ================================== */}

        <section className="students-grid">

          {
            loading
              ? (
                <p>
                  Cargando estudiantes...
                </p>
              )
              : filteredStudents.map(
                (student) => (

                  <article
                    key={student.id_usuario}
                    className="student-card"
                  >

                    <div className="student-top">

                      <div className="student-avatar">

                        {
                          student.nombres
                            .charAt(0)
                        }

                      </div>

                      <span className="status active">

                        Activo

                      </span>

                    </div>

                    <h3>

                      {
                        student.nombres
                      } {

                        student.apellidos
                      }

                    </h3>

                    <p>

                      {
                        student.grado || "Sin grado"
                      }

                      {" - Sección "}

                      {
                        student.seccion || "-"
                      }

                    </p>

                    {/* INFO */}

                    <div className="student-metrics">

                      <div>

                        <strong>
                          {
                            student.correo
                          }
                        </strong>

                        <span>
                          Correo
                        </span>

                      </div>

                      <div>

                        <strong>
                          {
                            student.celular || "-"
                          }
                        </strong>

                        <span>
                          Celular
                        </span>

                      </div>

                    </div>

                    <Link
                      to={`/estudiantes/${student.id_usuario}`}
                      className="student-btn"
                    >

                      Ver detalle

                    </Link>

                  </article>
                )
              )
          }

        </section>

      </div>

      {/* =================================
          MODAL
      ================================== */}

      {
        showModal && (

          <div className="create-student-overlay">

            <div className="create-student-modal">

              <div className="create-student-header">

                <h2>
                  Crear estudiante
                </h2>

                <button
                  className="create-student-close"
                  onClick={() =>
                    setShowModal(false)
                  }
                >
                  ✕
                </button>

              </div>

              <div className="create-student-form">

                <input
                  type="text"
                  name="nombres"
                  placeholder="Nombres"
                  value={formData.nombres}
                  onChange={handleChange}
                />

                <input
                  type="text"
                  name="apellidos"
                  placeholder="Apellidos"
                  value={formData.apellidos}
                  onChange={handleChange}
                />

                <input
                  type="date"
                  name="fecha_nacimiento"
                  value={formData.fecha_nacimiento}
                  onChange={handleChange}
                />

                <input
                  type="text"
                  name="grado"
                  placeholder="Grado"
                  value={formData.grado}
                  onChange={handleChange}
                />

                <input
                  type="text"
                  name="seccion"
                  placeholder="Sección"
                  value={formData.seccion}
                  onChange={handleChange}
                />

                <input
                  type="email"
                  name="correo"
                  placeholder="Correo"
                  value={formData.correo}
                  onChange={handleChange}
                />

                <input
                  type="text"
                  name="celular"
                  placeholder="Celular"
                  value={formData.celular}
                  onChange={handleChange}
                />

                <input
                  type="password"
                  name="password"
                  placeholder="Contraseña"
                  value={formData.password}
                  onChange={handleChange}
                />

                <button
                  className="create-student-submit"
                  onClick={createStudent}
                  disabled={saving}
                >

                  {
                    saving
                      ? "Guardando..."
                      : "Crear estudiante"
                  }

                </button>

              </div>

            </div>

          </div>
        )
      }

    </Layout>
  );
}

export default GestionEstudiantes;