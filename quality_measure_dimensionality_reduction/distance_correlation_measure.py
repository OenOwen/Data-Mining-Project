from scipy.spatial.distance import pdist
import numpy as np
from .base_quality_measure_dr import BaseQualityMeasureDR

class DistanceCorrelationMeasure(BaseQualityMeasureDR):
    def evaluate(self):
        dist_original = pdist(self.original_data)
        dist_reduced = pdist(self.reduced_data)
        return np.corrcoef(dist_original, dist_reduced)[0, 1]
