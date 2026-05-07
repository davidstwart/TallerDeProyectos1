from infrastructure.security.password_manager import (
    verify_password
)

from infrastructure.security.jwt_manager import (
    create_access_token
)

from application.ports.input.auth_input_port import (
    IAuthInputPort
)


class AuthUseCase(IAuthInputPort):

    def __init__(
        self,
        auth_repository,
        usuario_repository
    ):

        self.auth_repository = auth_repository

        self.usuario_repository = (
            usuario_repository
        )

    def login(self, data):

        user = (
            self.usuario_repository.find_by_email(
                data.correo
            )
        )

        if not user:
            raise Exception(
                "Usuario no encontrado"
            )

        valid_password = verify_password(
            data.password,
            user.password
        )

        if not valid_password:
            raise Exception(
                "Contraseña incorrecta"
            )

        access_token = create_access_token({
            "sub": user.correo,
            "rol": user.id_rol,
            "id_usuario": user.id_usuario
        })

        return {
            "access_token": access_token,
            "token_type": "bearer",

            "user": {
                "id_usuario": user.id_usuario,
                "nombres": user.nombres,
                "apellidos": user.apellidos,
                "correo": user.correo,
                "id_rol": user.id_rol
            }
        }

    def recover_password(self, data):

        return {
            "message": "Correo enviado"
        }

    def reset_password(self, data):

        return {
            "message": "Password actualizado"
        }