from fastapi import HTTPException


def verify_admin(user):

    if user["rol"] != "ADMINISTRADOR":
        raise HTTPException(
            status_code=403,
            detail="No autorizado"
        )


def verify_docente(user):

    if user["rol"] != "DOCENTE":
        raise HTTPException(
            status_code=403,
            detail="No autorizado"
        )


def verify_estudiante(user):

    if user["rol"] != "ESTUDIANTE":
        raise HTTPException(
            status_code=403,
            detail="No autorizado"
        )