from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from domain.models.user import User as DomainUser

DATABASE_URL = "mysql+pymysql://root:@localhost:3306/EduIA"
# si tienes password → root:admin

Base = declarative_base()


# ── Modelo DB ─────────────────────────────────────────
class UserORM(Base):
    __tablename__ = "usuario"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password = Column(String(255))
    rol = Column(String(50))


# ── Repository ────────────────────────────────────────
class MySQLUserRepository:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    # ✔️ coincide con use case
    def save(self, user: DomainUser):
        db = self.SessionLocal()
        try:
            user_db = UserORM(
                email=user.email,
                password=user.password,
                rol=user.rol,
            )
            db.add(user_db)
            db.commit()
            db.refresh(user_db)

            return DomainUser(
                user_db.id,
                user_db.email,
                user_db.password,
                user_db.rol,
            )
        finally:
            db.close()

    # ✔️ coincide con use case
    def find_by_email(self, email: str):
        db = self.SessionLocal()
        try:
            user = db.query(UserORM).filter(UserORM.email == email).first()

            if not user:
                return None

            return DomainUser(
                user.id,
                user.email,
                user.password,
                user.rol,
            )
        finally:
            db.close()