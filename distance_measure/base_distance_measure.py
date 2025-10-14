from abc import ABC, abstractmethod
import numpy as np

class BaseDistanceMeasure(ABC):

    @abstractmethod
    def compute(self, point1, point2):
        pass

    def _to_numpy(self, point1, point2):
        return np.array(point1), np.array(point2)
