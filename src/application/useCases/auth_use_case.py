from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from domain.models.user import User

# Configuración JWT
SECRET_KEY = "mi_llave_secreta_super_segura"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthUseCase:
    def __init__(self, repository):
        self.repository = repository

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def register(self, email, password, rol):
        hashed = pwd_context.hash(password)
        new_user = User(None, email, hashed, rol)
        user_saved = self.repository.save(new_user)
        
        # Generar token inmediatamente
        token = self.create_access_token({"sub": user_saved.email, "rol": user_saved.rol})
        return {"access_token": token, "token_type": "bearer", "user": {"email": user_saved.email, "rol": user_saved.rol}}

    def login(self, email, password):
        user = self.repository.find_by_email(email)
        if user and pwd_context.verify(password, user.password):
            token = self.create_access_token({"sub": user.email, "rol": user.rol})
            return {"access_token": token, "token_type": "bearer", "user": {"email": user.email, "rol": user.rol}}
        return None