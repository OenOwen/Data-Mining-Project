from abc import ABC, abstractmethod
import numpy as np

class BaseQualityMeasure(ABC):

    def __init__(self, dataset, labels):
        self.data = dataset.getData()
        self.labels = np.array(labels)

    def _valid_labels(self):
        return len(np.unique(self.labels)) >= 2

    @abstractmethod
    def evaluate(self):
        pass
