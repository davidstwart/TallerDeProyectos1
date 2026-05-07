import hashlib


def generate_sha256(content: bytes):

    return hashlib.sha256(
        content
    ).hexdigest()