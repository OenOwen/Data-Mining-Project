import numpy as np
from .base_distance_measure import BaseDistanceMeasure

class ManhattanDistance(BaseDistanceMeasure):
    def compute(self, point1, point2):
        point1, point2 = self._to_numpy(point1, point2)
        return np.sum(np.abs(point1 - point2))
