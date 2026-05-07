from infrastructure.database.database import (
    SessionLocal
)

from application.ports.output.auth_output_port import (
    IAuthOutputPort
)


class MySQLAuthRepository(
    IAuthOutputPort
):

    def __init__(self):

        self.db = SessionLocal()

    def save_code(
        self,
        auth
    ):

        pass

    def verify_code(
        self,
        correo: str,
        codigo: str
    ):

        pass