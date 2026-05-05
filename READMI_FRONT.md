# Frontend React — Guía de inicio

El frontend de este proyecto está construido con **React 19 + Vite** y se encuentra en:

```
infrastructure/frameworks/react/
```

---

## Requisitos previos

- [Node.js](https://nodejs.org/) v18 o superior
- npm (incluido con Node.js)

---

## Pasos para levantar el proyecto

### 1. Navegar al directorio del frontend

```bash
cd infrastructure/frameworks/react
```

### 2. Instalar dependencias

```bash
npm install
```

### 3. Iniciar el servidor de desarrollo

```bash
npm run dev
```

Vite levantará el servidor en:

```
http://localhost:5173
```

---

## Scripts disponibles

| Comando | Descripción |
|---|---|
| `npm run dev` | Inicia el servidor de desarrollo con hot-reload |
| `npm run build` | Genera el bundle de producción en `dist/` |
| `npm run preview` | Sirve el build de producción localmente |
| `npm run lint` | Ejecuta ESLint sobre el código fuente |

---

## Estructura del frontend

```
infrastructure/frameworks/react/
├── index.html
├── vite.config.js
├── eslint.config.js
├── package.json
├── public/
└── src/
    ├── main.jsx          # Punto de entrada
    ├── App.jsx
    ├── assets/
    ├── componentes/
    │   └── Navbar.jsx
    ├── estilos/
    │   ├── auth.css
    │   ├── global.css
    │   └── pages.css
    └── paginas/
        ├── Home.jsx
        ├── Login.jsx
        ├── Registro.jsx
        ├── Cursos.jsx
        ├── Laboratorio.jsx
        ├── Progreso.jsx
        └── recuperarpass.jsx
```

---

## Dependencias principales

| Paquete | Versión |
|---|---|
| react | ^19.2.5 |
| react-dom | ^19.2.5 |
| react-router-dom | ^7.14.2 |
| vite | ^8.0.10 |
