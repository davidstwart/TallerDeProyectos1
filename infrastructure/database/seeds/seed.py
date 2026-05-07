from infrastructure.database.database import (
    SessionLocal
)

from infrastructure.adapters.output.orm.rol_orm import (
    RolORM
)

from infrastructure.adapters.output.orm.usuario_orm import (
    UsuarioORM
)

from infrastructure.security.password_manager import (
    hash_password
)

# =====================================
# DB SESSION
# =====================================

db = SessionLocal()

# =====================================
# ROLES
# =====================================

roles = [

    {
        "id_rol": 1,
        "nombre": "Administrador",
    },

    {
        "id_rol": 2,
        "nombre": "Docente",
    },

    {
        "id_rol": 3,
        "nombre": "Estudiante",
    },
]

for role in roles:

    exists = (

        db.query(RolORM)

        .filter(
            RolORM.id_rol ==
            role["id_rol"]
        )

        .first()
    )

    if not exists:

        db.add(

            RolORM(
                id_rol=role["id_rol"],
                nombre=role["nombre"]
            )
        )

# =====================================
# ADMIN
# =====================================

admin_exists = (

    db.query(UsuarioORM)

    .filter(
        UsuarioORM.correo ==
        "admin@eduia.com"
    )

    .first()
)

if not admin_exists:

    admin = UsuarioORM(

        nombres="Admin",

        apellidos="Sistema",

        correo="admin@eduia.com",

        celular="999999999",

        password=
            hash_password("123456"),

        id_rol=1,

        activo=True,
    )

    db.add(admin)

# =====================================
# DOCENTE
# =====================================

docente_exists = (

    db.query(UsuarioORM)

    .filter(
        UsuarioORM.correo ==
        "docente@eduia.com"
    )

    .first()
)

if not docente_exists:

    docente = UsuarioORM(

        nombres="Carlos",

        apellidos="Docente",

        correo="docente@eduia.com",

        celular="988888888",

        password=
            hash_password("123456"),

        id_rol=2,

        activo=True,
    )

    db.add(docente)

# =====================================
# ESTUDIANTE
# =====================================

student_exists = (

    db.query(UsuarioORM)

    .filter(
        UsuarioORM.correo ==
        "estudiante@eduia.com"
    )

    .first()
)

if not student_exists:

    student = UsuarioORM(

        nombres="Rosalyn",

        apellidos="Estudiante",

        correo="estudiante@eduia.com",

        celular="977777777",

        grado="5to",

        seccion="A",

        password=
            hash_password("123456"),

        id_rol=3,

        activo=True,
    )

    db.add(student)

# =====================================
# COMMIT
# =====================================

db.commit()

# =====================================
# CLOSE
# =====================================

db.close()

print(
    "Seed ejecutado correctamente"
)