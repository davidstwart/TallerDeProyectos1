import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

import Home from "./paginas/Home";
import Login from "./paginas/Login";
import Registro from "./paginas/Registro";
import Cursos from "./paginas/Cursos";
import Laboratorio from "./paginas/Laboratorio";
import Progreso from "./paginas/Progreso";
import PanelDocente from "./paginas/PanelDocente";
import GestionEstudiantes from "./paginas/GestionEstudiantes";
import CursoDetalle from "./paginas/CursoDetalle";
import EstudianteDetalle from "./paginas/EstudianteDetalle";
import LaboratorioEstudiante from "./paginas/LaboratorioEstudiante";
import AdminDocentes from "./paginas/AdminDocentes";
import ProtectedRoute from "./routes/ProtectedRoute";
import RoleGuard from "./routes/RoleGuard";

function App() {

  return (

    <BrowserRouter>

      <Routes>


        {/* =========================
            PUBLIC
        ========================== */}

        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/login"
          element={<Login />}
        />

        <Route
          path="/registro"
          element={<Registro />}
        />

        {/* =========================
            ADMIN
        ========================= */}

        <Route
          path="/admin"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[1]}
              >

                <AdminDocentes />

              </RoleGuard>

            </ProtectedRoute>
          }
        />



        {/* =========================
            DOCENTE
        ========================== */}

        <Route
          path="/docente"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[2]}
              >

                <PanelDocente />

              </RoleGuard>

            </ProtectedRoute>
          }
        />
        <Route
          path="/laboratorio"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[
                  2,
                ]}
              >

                <Laboratorio />

              </RoleGuard>

            </ProtectedRoute>
          }
        />

        {/* =========================
            ESTUDIANTE
        ========================== */}

        <Route
          path="/estudiantes"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[2]}
              >

                <GestionEstudiantes />

              </RoleGuard>

            </ProtectedRoute>
          }
        />

        <Route
          path="/estudiantes/:id"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[2]}
              >

                <EstudianteDetalle />

              </RoleGuard>

            </ProtectedRoute>
          }
        />

        <Route
          path="/laboratorio-estudiante"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[3]}
              >

                <LaboratorioEstudiante />

              </RoleGuard>

            </ProtectedRoute>
          }
        />

        <Route
          path="/progreso"
          element={

            <ProtectedRoute>

              <RoleGuard
                allowedRoles={[
                  3,
                ]}
              >

                <Progreso />

              </RoleGuard>

            </ProtectedRoute>
          }
        />

        {/* =========================
            TODOS AUTH
        ========================== */}

        <Route
          path="/cursos"
          element={

            <ProtectedRoute>

              <Cursos />

            </ProtectedRoute>
          }
        />

        <Route
          path="/curso/:id"
          element={

            <ProtectedRoute>

              <CursoDetalle />

            </ProtectedRoute>
          }
        />

      </Routes>

    </BrowserRouter>
  );
}

export default App;