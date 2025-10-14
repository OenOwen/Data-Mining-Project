from abc import ABC, abstractmethod
import numpy as np

class BaseQualityMeasureDR(ABC):
    def __init__(self, original_data, reduced_data, n_neighbors=5):
        self.original_data = np.array(original_data)
        self.reduced_data = np.array(reduced_data)
        self.n_neighbors = n_neighbors

    @abstractmethod
    def evaluate(self):
        pass
