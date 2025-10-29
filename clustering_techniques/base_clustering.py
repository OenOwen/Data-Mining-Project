from abc import ABC, abstractmethod

class BaseClustering(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def fit(self):
        pass