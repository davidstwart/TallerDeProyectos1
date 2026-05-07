from abc import ABC, abstractmethod


class IDatasetInputPort(ABC):

    @abstractmethod
    def upload_dataset(self, file, usuario_id: int):
        pass