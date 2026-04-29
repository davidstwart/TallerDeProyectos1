import { Link } from "react-router-dom";
import "../estilos/auth.css";

function Registro() {
  return (
    <div className="auth-page">
      <div className="auth-container register-layout">
        <section className="auth-card">
          <Link to="/login" className="auth-back">
            ← Volver al login
          </Link>

          <div className="auth-header">
            <div className="auth-logo">IA</div>

            <div>
              <h1>Crear cuenta</h1>
              <p>Regístrate para acceder a EDU IA.</p>
            </div>
          </div>

          <form className="auth-form">
            <div className="form-row">
              <div className="form-group">
                <label>Nombres</label>
                <input type="text" placeholder="Ingresa tus nombres" />
              </div>

              <div className="form-group">
                <label>Apellidos</label>
                <input type="text" placeholder="Ingresa tus apellidos" />
              </div>
            </div>

            <div className="form-group">
              <label>Correo electrónico</label>
              <input type="email" placeholder="ejemplo@correo.com" />
            </div>

            <div className="form-group">
              <label>Contraseña</label>
              <input type="password" placeholder="Crea una contraseña" />
            </div>

            <div className="form-group">
              <label>Confirmar contraseña</label>
              <input type="password" placeholder="Repite tu contraseña" />
            </div>

            <div className="form-group">
              <label>Tipo de usuario</label>
              <select>
                <option>Estudiante</option>
                <option>Docente</option>
                <option>Administrador</option>
              </select>
            </div>

            <label className="auth-check">
              <input type="checkbox" />
              <span>Acepto los términos y condiciones de la plataforma.</span>
            </label>

            <button type="submit" className="auth-submit">
              Crear cuenta
            </button>
          </form>

          <p className="auth-footer">
            ¿Ya tienes cuenta? <Link to="/login">Inicia sesión</Link>
          </p>
        </section>

        <section className="auth-info">
          <span className="auth-badge">EDU IA</span>

          <h2>Empieza tu aprendizaje en Inteligencia Artificial</h2>

          <p>
            Crea una cuenta para acceder a cursos, evaluaciones, laboratorio IA,
            seguimiento de progreso e insignias digitales.
          </p>

          <div className="auth-benefits">
            <article>
              <span>📚</span>
              <div>
                <h3>Cursos interactivos</h3>
                <p>Aprende conceptos de IA de forma organizada.</p>
              </div>
            </article>

            <article>
              <span>📝</span>
              <div>
                <h3>Evaluaciones</h3>
                <p>Mide tu aprendizaje mediante actividades y pruebas.</p>
              </div>
            </article>

            <article>
              <span>🏅</span>
              <div>
                <h3>Insignias</h3>
                <p>Obtén logros por completar módulos y retos.</p>
              </div>
            </article>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Registro;