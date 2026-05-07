from abc import ABC, abstractmethod


class ISessionRepository(ABC):

    @abstractmethod
    def save_dataframe(self, session_id, df, target_column):
        pass

    @abstractmethod
    def get_dataframe(self, session_id):
        pass

    @abstractmethod
    def save_trained_model(
        self,
        session_id,
        model,
        scaler,
        feature_names,
        model_name
    ):
        pass