from abc import ABC, abstractmethod


class IEmailOutputPort(ABC):

    @abstractmethod
    def send_email(
        self,
        to: str,
        subject: str,
        body: str
    ):
        pass