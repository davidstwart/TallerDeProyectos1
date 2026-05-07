import {
  Link,
} from "react-router-dom";

import "../estilos/navbar.css";

function Sidebar() {

  return (
    <aside className="dashboard-sidebar">

      <div className="navbar-brand">

        <div className="navbar-logo">
          IA
        </div>

        <h2>EDU IA</h2>
      </div>

      <nav
        className="sidebar-links"
        style={{
          marginTop: "32px",
          display: "grid",
          gap: "18px",
        }}
      >

        <Link to="/dashboard">
          Dashboard
        </Link>

        <Link to="/laboratorio">
          Laboratorio
        </Link>

        <Link to="/datasets">
          Datasets
        </Link>

        <Link to="/progreso">
          Progreso
        </Link>

      </nav>

    </aside>
  );
}

export default Sidebar;