import {
  Link,
  useNavigate,
} from "react-router-dom";

import {
  useState,
} from "react";

import {
  loginRequest,
} from "../services/authService";

import {
  useAuth,
} from "../context/AuthContext";

import "../estilos/auth.css";

function Login() {

  const navigate =
    useNavigate();

  const { login } =
    useAuth();

  const [formData, setFormData] =
    useState({
      correo: "",
      password: "",
    });

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const handleChange = (e) => {

    setFormData({
      ...formData,
      [e.target.name]:
        e.target.value,
    });
  };

  const iniciarSesion = async (
    e
  ) => {

    e.preventDefault();

    setError("");

    try {

      setLoading(true);

      const response =
        await loginRequest(
          formData
        );

      login(
        response.access_token,
        response.user
      );

      // =================================
      // REDIRECCIÓN POR ROL
      // =================================

      const rol =
        response.user.id_rol;

      if (rol === 1) {

        navigate("/admin");

      } else if (rol === 2) {

        navigate("/docente");

      } else {

        navigate("/progreso");
      }

    } catch (error) {

      console.error(error);

      setError(
        "Correo o contraseña incorrectos"
      );

    } finally {

      setLoading(false);
    }
  };

  return (

    <div className="login-page">

      <div className="login-container">

        <section className="login-info">

          <div className="login-brand">

            <div className="login-logo">
              IA
            </div>

            <span>EDU IA</span>
          </div>

          <h1>
            Bienvenido de nuevo
          </h1>

          <p>
            Inicia sesión para acceder
            a cursos, laboratorio IA,
            progreso académico y
            modelos entrenados.
          </p>

        </section>

        <section className="login-card">

          <Link
            to="/"
            className="login-back"
          >
            ← Volver al inicio
          </Link>

          <h2>
            Iniciar sesión
          </h2>

          <p className="login-subtitle">
            Ingresa tus credenciales.
          </p>

          {
            error && (
              <div className="form-error">
                {error}
              </div>
            )
          }

          <form
            className="login-form"
            onSubmit={iniciarSesion}
          >

            <div className="form-group">

              <label>
                Correo electrónico
              </label>

              <input
                type="email"
                name="correo"
                value={formData.correo}
                onChange={handleChange}
                placeholder="ejemplo@correo.com"
                required
              />
            </div>

            <div className="form-group">

              <label>
                Contraseña
              </label>

              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Ingresa tu contraseña"
                required
              />
            </div>

            <button
              type="submit"
              className="login-submit"
              disabled={loading}
            >

              {
                loading
                  ? "Ingresando..."
                  : "Ingresar"
              }

            </button>

          </form>

          <p className="login-register">

            ¿No tienes cuenta?

            <Link to="/registro">
              Regístrate aquí
            </Link>

          </p>

        </section>

      </div>

    </div>
  );
}

export default Login;