# src/application/ports/output/user_repository.py
from abc import ABC, abstractmethod
from domain.models.user import User

class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> User:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> User | None:
        pass