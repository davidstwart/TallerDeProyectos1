import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

import "../../estilos/layout.css";

function Layout({
  children,
}) {

  return (

    <div className="layout">

      {/* SIDEBAR */}

      <Sidebar />

      {/* MAIN */}

      <div className="layout-main">

        {/* TOPBAR */}

        <Topbar />

        {/* CONTENT */}

        <main className="layout-content">

          {children}

        </main>

      </div>

    </div>
  );
}

export default Layout;