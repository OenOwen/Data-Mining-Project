from abc import ABC, abstractmethod

class BaseDimensionalityReduction(ABC):

    def __init__(self, dataset, n_components=2):
        self.dataset = dataset
        self.n_components = n_components

    @abstractmethod
    def reduce(self):
        pass
