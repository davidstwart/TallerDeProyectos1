from application.ports.input.machine_learning_input_port import (
    IMachineLearningInputPort
)


class MachineLearningUseCase(
    IMachineLearningInputPort
):

    def __init__(
        self,
        ml_repository,
        session_repository
    ):
        self.ml_repository = ml_repository
        self.session_repository = session_repository

    def train_model(self, data):

        return {
            "message": "Modelo entrenado"
        }

    def predict(self, data):

        return {
            "message": "Predicción realizada"
        }