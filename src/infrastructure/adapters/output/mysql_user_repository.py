from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from domain.models.user import User as DomainUser

Base = declarative_base()

class UserTable(Base):
    __tablename__ = "usuario"
    id_usuario = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), unique=True)
    password = Column(String(255))
    rol = Column(String(20))

class MySQLUserRepository:
    def __init__(self):
        # AJUSTA AQUÍ: usuario:contraseña@localhost/nombre_bd
        self.engine = create_engine("mysql+pymysql://root:admin@localhost:3306/EduIA")
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save(self, user: DomainUser):
        session = self.SessionLocal()
        db_user = UserTable(email=user.email, password=user.password, rol=user.rol)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        user.id_usuario = db_user.id_usuario
        session.close()
        return user

    def find_by_email(self, email: str):
        session = self.SessionLocal()
        user_db = session.query(UserTable).filter(UserTable.email == email).first()
        session.close()
        if user_db:
            return DomainUser(user_db.id_usuario, user_db.email, user_db.password, user_db.rol)
        return None