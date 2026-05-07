from infrastructure.security.password_manager import (
    hash_password
)

password = "123456"

hashed = hash_password(password)

print(hashed)