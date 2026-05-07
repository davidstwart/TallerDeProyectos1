import {
  useNavigate,
} from "react-router-dom";

import {
  useAuth,
} from "../../context/AuthContext";

function Topbar() {

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

  return (

    <header className="topbar">

      <div>

        <h1>
          Bienvenido
        </h1>

        <p>
          Gestiona tu aprendizaje
          y laboratorio IA.
        </p>

      </div>

      <div className="topbar-user">

        <div className="topbar-avatar">

          {
            user?.nombres?.charAt(0)
          }

        </div>

        <div>

          <strong>
            {user?.nombres}
          </strong>

          <span>

            {
              user?.id_rol === 1
                ? "Administrador"

                : user?.id_rol === 2
                ? "Docente"

                : "Estudiante"
            }

          </span>

        </div>

        <button
          onClick={handleLogout}
          className="logout-btn"
        >

          Salir

        </button>

      </div>

    </header>
  );
}

export default Topbar;