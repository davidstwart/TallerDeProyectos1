from infrastructure.database.database import (
    SessionLocal
)

from application.ports.output.machine_learning_output_port import (
    IMachineLearningOutputPort
)


class MachineLearningRepository(
    IMachineLearningOutputPort
):

    def __init__(self):

        self.db = SessionLocal()

    def save_model(
        self,
        model_data
    ):

        pass

    def load_model(
        self,
        model_id: int
    ):

        pass