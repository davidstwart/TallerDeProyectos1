from fastapi import APIRouter

router = APIRouter(
    prefix="/roles",
    tags=["Roles"]
)


@router.get("/")
def get_roles():

    return [
        {
            "id_rol": 1,
            "nombre": "ADMINISTRADOR"
        },
        {
            "id_rol": 2,
            "nombre": "DOCENTE"
        },
        {
            "id_rol": 3,
            "nombre": "ESTUDIANTE"
        }
    ]