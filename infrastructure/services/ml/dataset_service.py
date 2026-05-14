import os
from fastapi import UploadFile


DATASET_DIR = "datasets"


def save_dataset_file(
    file: UploadFile,
    content: bytes
):

    os.makedirs(DATASET_DIR, exist_ok=True)

    file_path = os.path.join(
        DATASET_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path