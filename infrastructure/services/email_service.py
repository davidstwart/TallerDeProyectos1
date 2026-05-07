import smtplib
from email.mime.text import MIMEText


class EmailService:

    def send_email(
        self,
        to: str,
        subject: str,
        body: str
    ):

        sender_email = "tu_correo@gmail.com"
        sender_password = "tu_password"

        msg = MIMEText(body)

        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        try:

            with smtplib.SMTP(
                "smtp.gmail.com",
                587
            ) as server:

                server.starttls()

                server.login(
                    sender_email,
                    sender_password
                )

                server.send_message(msg)

            return True

        except Exception as e:
            print(e)
            return False