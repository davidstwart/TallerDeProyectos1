import { BrowserRouter, Routes, Route } from "react-router-dom";

import Home from "./paginas/Home";
import Login from "./paginas/Login";
import Registro from "./paginas/Registro";
import Cursos from "./paginas/Cursos";
import Laboratorio from "./paginas/Laboratorio";
import Progreso from "./paginas/Progreso";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/registro" element={<Registro />} />
        <Route path="/cursos" element={<Cursos />} />
        <Route path="/laboratorio" element={<Laboratorio />} />
        <Route path="/progreso" element={<Progreso />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;