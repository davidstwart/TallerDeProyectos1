import {
  Link,
  useLocation,
} from "react-router-dom";

import {
  useAuth,
} from "../../context/AuthContext";

function Sidebar() {

  const location =
    useLocation();

  const {
    user,
  } = useAuth();

  // =====================================
  // LINKS POR ROL
  // =====================================

  const docenteLinks = [

    {
      path: "/laboratorio",
      label: "Laboratorio IA",
    },

    {
      path: "/cursos",
      label: "Cursos",
    },

    {
      path: "/estudiantes",
      label: "Estudiantes",
    },

  ];

  const estudianteLinks = [

    {
      path: "/cursos",
      label: "Mis Cursos",
    },

    {
      path: "/progreso",
      label: "Mi Progreso",
    },

    {
      path: "/laboratorio-estudiante",
      label: "Laboratorio",
    },
  ];

  const adminLinks = [

    {
      path: "/admin",
      label: "Docentes",
    },
  ];

  // eslint-disable-next-line no-useless-assignment
  let links = [];

  // ADMIN
  if (user?.id_rol === 1) {

    links = adminLinks;
  }

  // DOCENTE
  else if (user?.id_rol === 2) {

    links = docenteLinks;
  }

  // ESTUDIANTE
  else {

    links = estudianteLinks;
  }

  return (

    <aside className="sidebar">

      {/* LOGO */}

      <div className="sidebar-logo">

        <div className="sidebar-logo-icon">
          IA
        </div>

        <div>

          <h2>
            EDU IA
          </h2>

          <span>
            Plataforma educativa
          </span>

        </div>

      </div>

      {/* LINKS */}

      <nav className="sidebar-nav">

        {
          links.map((link) => (

            <Link
              key={link.path}
              to={link.path}
              className={

                location.pathname ===
                link.path

                  ? "sidebar-link active"

                  : "sidebar-link"
              }
            >

              {link.label}

            </Link>
          ))
        }

      </nav>

    </aside>
  );
}

export default Sidebar;