from abc import ABC, abstractmethod


class IDatasetOutputPort(ABC):

    @abstractmethod
    def save(self, dataset):
        pass

    @abstractmethod
    def find_by_hash(self, hash_dataset: str):
        pass

    @abstractmethod
    def find_by_name(self, nombre: str):
        pass