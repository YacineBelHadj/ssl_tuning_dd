from abc import ABC, abstractmethod
from typing import List, Tuple

class DDSystem(ABC):
    def __init__(self,encoder) -> None:
        self.encoder = encoder
        self.system = None
    @abstractmethod
    def fit(self,data):
        pass
    @abstractmethod
    def predict(self,data):
        pass
    @abstractmethod
    def save(self,path):
        pass
    @abstractmethod
    def load(self,path):
        pass


