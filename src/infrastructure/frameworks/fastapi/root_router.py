from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Root"])


@router.get("/", response_class=HTMLResponse)
def landing():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IA Lab</title>
        <style>
            body {
                margin: 0;
                font-family: Arial;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                text-align: center;
            }
            h1 {
                font-size: 3rem;
            }
            p {
                font-size: 1.2rem;
            }
            a {
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: white;
                color: #1e3c72;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🚀 Laboratorio de IA</h1>
            <p>Entrena modelos, genera datasets y predice resultados.</p>
            <a href="/docs">Ir a Swagger</a>
        </div>
    </body>
    </html>
    """