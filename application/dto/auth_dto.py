from pydantic import BaseModel, EmailStr


class LoginDTO(BaseModel):
    correo: EmailStr
    password: str


class RecoverPasswordDTO(BaseModel):
    correo: EmailStr


class ResetPasswordDTO(BaseModel):
    correo: EmailStr
    codigo: str
    nueva_password: str


class VerifyCodeDTO(BaseModel):
    correo: EmailStr
    codigo: str