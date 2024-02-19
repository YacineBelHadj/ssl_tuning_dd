from abc import ABC, abstractmethod

class BaseDDS(ABC):
    def __init__(self) -> None:
        super().__init__()
        # Initialize any necessary attributes or leave empty if none are needed

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict_samples(self, data):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
