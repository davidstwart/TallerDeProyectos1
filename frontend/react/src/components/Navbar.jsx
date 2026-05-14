import {
  Link,
  useNavigate,
} from "react-router-dom";

import {
  useAuth,
} from "../context/AuthContext";

import "../estilos/navbar.css";

function Navbar() {

  const navigate =
    useNavigate();

  const {
    user,
    logout,
  } = useAuth();

  const handleLogout = () => {

    logout();

    navigate("/login");
  };

  const getRoleName = () => {

    switch (user?.id_rol) {

      case 1:
        return "Administrador";

      case 2:
        return "Docente";

      case 3:
        return "Estudiante";

      default:
        return "Usuario";
    }
  };

  return (

    <header className="navbar">

      {/* =========================
          BRAND
      ========================== */}

      <div className="navbar-brand">

        <div className="navbar-logo">
          IA
        </div>

        <div>

          <h2>EDU IA</h2>

          <span className="navbar-subtitle">
            Plataforma Educativa
          </span>

        </div>
      </div>

      {/* =========================
          LINKS
      ========================== */}

      <nav className="navbar-links">

        <Link to="/">
          Inicio
        </Link>

        <Link to="/cursos">
          Cursos
        </Link>

        {/* =========================
            ADMIN
        ========================== */}

        {
          user?.id_rol === 1 && (

            <Link to="/admin">
              Administración
            </Link>
          )
        }

        {/* =========================
            DOCENTE
        ========================== */}

        {
          (
            user?.id_rol === 1 ||
            user?.id_rol === 2
          ) && (

            <Link to="/laboratorio">
              Laboratorio IA
            </Link>
          )
        }

        {/* =========================
            ESTUDIANTE
        ========================== */}

        {
          user?.id_rol === 3 && (

            <Link to="/progreso">
              Mi progreso
            </Link>
          )
        }

      </nav>

      {/* =========================
          USER
      ========================== */}

      <div className="navbar-user">

        <div className="navbar-user-info">

          <div className="navbar-avatar">

            {
              user?.nombres?.charAt(0)
            }

          </div>

          <div>

            <strong>
              {user?.nombres}
            </strong>

            <span className="navbar-role">

              {getRoleName()}

            </span>

          </div>

        </div>

        <button
          className="btn-secondary"
          onClick={handleLogout}
        >
          Salir
        </button>

      </div>

    </header>
  );
}

export default Navbar;