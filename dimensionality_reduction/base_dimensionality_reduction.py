from abc import ABC, abstractmethod

class BaseDimensionalityReduction(ABC):

    def __init__(self, dataset, n_components=2, distance_measure='euclidean'):
        self.dataset = dataset
        self.n_components = n_components
        self.distance_measure = distance_measure

    @abstractmethod
    def reduce(self):
        pass
