import { Link } from "react-router-dom";
import "../App.css";

function Home() {
  return (
    <div className="home-page">
      <header className="home-navbar">
        <div className="home-brand">
          <div className="home-logo">IA</div>
          <span>EDU IA</span>
        </div>

        <nav className="home-menu">
        </nav>

        <Link to="/login" className="home-login-btn">
          Iniciar sesión
        </Link>
      </header>

      <main className="home-hero">
        <section className="home-hero-text">
          <span className="home-badge">Plataforma educativa inteligente</span>

          <h1>Aprende Inteligencia Artificial de forma práctica</h1>

          <p>
            EDU IA es una plataforma web donde los estudiantes podrán aprender
            conceptos de inteligencia artificial mediante cursos, evaluaciones,
            seguimiento de progreso, insignias y un laboratorio interactivo.
          </p>

          <div className="home-actions">
            <Link to="/login" className="home-primary-btn">
              Iniciar sesión
            </Link>

            <a href="#funciones" className="home-secondary-btn">
              Ver funciones
            </a>
          </div>
        </section>

        <section className="home-preview-card">
          <div className="preview-header">
            <div>
              <span>Vista previa</span>
              <h3>Panel del estudiante</h3>
            </div>
          </div>

          <div className="preview-grid">
            <div className="preview-item">
              <span>📚</span>
              <p>Cursos</p>
            </div>

            <div className="preview-item">
              <span>📝</span>
              <p>Exámenes</p>
            </div>

            <div className="preview-item">
              <span>🧪</span>
              <p>Laboratorio IA</p>
            </div>

            <div className="preview-item">
              <span>🏅</span>
              <p>Insignias</p>
            </div>
          </div>
        </section>
      </main>

      <section className="home-section" id="funciones">
        <div className="section-title">
          <span>¿Qué encontrarás al iniciar sesión?</span>
          <h2>Funciones principales de la plataforma</h2>
          <p>
            Al acceder a tu cuenta, podrás usar diferentes módulos diseñados
            para aprender, practicar y medir tu avance.
          </p>
        </div>

        <div className="home-features">
          <article>
            <span>📚</span>
            <h3>Cursos interactivos</h3>
            <p>
              Accede a lecciones organizadas por módulos para aprender los
              fundamentos de la inteligencia artificial.
            </p>
          </article>

          <article>
            <span>📝</span>
            <h3>Evaluaciones</h3>
            <p>
              Resuelve exámenes y actividades para comprobar tu comprensión.
            </p>
          </article>

          <article>
            <span>📈</span>
            <h3>Seguimiento de progreso</h3>
            <p>
              Consulta tu avance, módulos completados y resultados obtenidos.
            </p>
          </article>

          <article>
            <span>🧪</span>
            <h3>Laboratorio IA</h3>
            <p>
              Experimenta con ejemplos prácticos sobre modelos de inteligencia
              artificial.
            </p>
          </article>

          <article>
            <span>🏅</span>
            <h3>Insignias</h3>
            <p>Gana reconocimientos digitales al completar actividades.</p>
          </article>

          <article>
            <span>🤖</span>
            <h3>Aprendizaje guiado</h3>
            <p>Recibe apoyo mediante recursos educativos sobre IA.</p>
          </article>
        </div>
      </section>

      <section className="home-modules" id="modulos">
        <div className="module-content">
          <span>Recorrido de aprendizaje</span>
          <h2>Desde conceptos básicos hasta práctica con modelos</h2>
          <p>
            La plataforma permite avanzar de manera progresiva: aprender,
            practicar, evaluar y mejorar.
          </p>
        </div>

        <div className="module-list">
          <div className="module-step">
            <strong>01</strong>
            <div>
              <h3>Aprender</h3>
              <p>Revisar cursos y contenidos sobre IA.</p>
            </div>
          </div>

          <div className="module-step">
            <strong>02</strong>
            <div>
              <h3>Practicar</h3>
              <p>Usar actividades y ejemplos interactivos.</p>
            </div>
          </div>

          <div className="module-step">
            <strong>03</strong>
            <div>
              <h3>Evaluar</h3>
              <p>Resolver pruebas para medir el aprendizaje.</p>
            </div>
          </div>

          <div className="module-step">
            <strong>04</strong>
            <div>
              <h3>Mejorar</h3>
              <p>Revisar resultados, progreso e insignias obtenidas.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-cta" id="beneficios">
        <h2>Empieza a explorar la plataforma educativa</h2>
        <p>
          Inicia sesión para acceder a los módulos, cursos, laboratorio y
          progreso.
        </p>

        <Link to="/login" className="home-primary-btn">
          Iniciar Sesion o Crear Cuenta
        </Link>
      </section>
    </div>
  );
}

export default Home;