""" from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
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
        self.engine = create_engine(
            "mysql+pymysql://root:admin@localhost:3306/EduIA",
            echo=False,  # opcional: True para debug SQL
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save(self, user: DomainUser):
        session = self.SessionLocal()
        try:
            db_user = UserTable(
                email=user.email,
                password=user.password,
                rol=user.rol,
            )
            session.add(db_user)
            session.commit()
            session.refresh(db_user)

            return DomainUser(
                db_user.id_usuario,
                db_user.email,
                db_user.password,
                db_user.rol,
            )

        except Exception as e:
            session.rollback()
            raise e

        finally:
            session.close()

    def find_by_email(self, email: str):
        session = self.SessionLocal()
        try:
            user_db = (
                session.query(UserTable)
                .filter(UserTable.email == email)
                .first()
            )

            if not user_db:
                return None

            return DomainUser(
                user_db.id_usuario,
                user_db.email,
                user_db.password,
                user_db.rol,
            )

        finally:
            session.close() """