# src/infrastructure/adapters/output/mysql_adapter.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from application.ports.output.user_repository import UserRepository
from domain.models.user import User as DomainUser

DATABASE_URL = "mysql+pymysql://usuario:password@localhost:3306/nombre_bd"

Base = declarative_base()

class UserTable(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password = Column(String(255))

class MySQLUserRepository(UserRepository):
    def __init__(self):
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(bind=engine)
        self.SessionLocal = sessionmaker(bind=engine)

    def save(self, user: DomainUser) -> DomainUser:
        db = self.SessionLocal()
        db_user = UserTable(email=user.email, password=user.password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        user.id = db_user.id
        db.close()
        return user

    def find_by_email(self, email: str) -> DomainUser | None:
        db = self.SessionLocal()
        user_data = db.query(UserTable).filter(UserTable.email == email).first()
        db.close()
        if user_data:
            return DomainUser(id=user_data.id, email=user_data.email, password=user_data.password)
        return None