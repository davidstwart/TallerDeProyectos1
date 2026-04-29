import { Link } from "react-router-dom";
import "../App.css";

function Login() {
  return (
    <div className="login-page">
      <div className="login-container">
        <section className="login-info">
          <div className="login-brand">
            <div className="login-logo">IA</div>
            <span>EDU IA</span>
          </div>

          <h1>Bienvenido de nuevo</h1>

          <p>
            Inicia sesión para acceder a tus cursos, evaluaciones, laboratorio
            de inteligencia artificial, progreso académico e insignias.
          </p>

          <div className="login-benefits">
            <div>
              <span>📚</span>
              <p>Cursos interactivos</p>
            </div>

            <div>
              <span>🧪</span>
              <p>Laboratorio IA</p>
            </div>

            <div>
              <span>📈</span>
              <p>Seguimiento de progreso</p>
            </div>
          </div>
        </section>

        <section className="login-card">
          <Link to="/" className="login-back">
            ← Volver al inicio
          </Link>

          <h2>Iniciar sesión</h2>
          <p className="login-subtitle">
            Ingresa tus datos para continuar en la plataforma.
          </p>

          <form className="login-form">
            <div className="form-group">
              <label>Correo electrónico</label>
              <input type="email" placeholder="ejemplo@correo.com" />
            </div>

            <div className="form-group">
              <label>Contraseña</label>
              <input type="password" placeholder="Ingresa tu contraseña" />
            </div>

            <div className="login-options">
              <label className="remember">
                <input type="checkbox" />
                <span>Recordarme</span>
              </label>

              <a href="#">¿Olvidaste tu contraseña?</a>
            </div>

            <button type="submit" className="login-submit">
              Ingresar
            </button>
          </form>

<p className="login-register">
  ¿No tienes cuenta? <Link to="/registro">Regístrate aquí</Link>
</p>
        </section>
      </div>
    </div>
  );
}

export default Login;