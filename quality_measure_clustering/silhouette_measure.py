from sklearn.metrics import silhouette_score
from .base_quality_measure import BaseQualityMeasure

class SilhouetteMeasure(BaseQualityMeasure):
    def evaluate(self):
        if not self._valid_labels():
            return float('nan')
        return silhouette_score(self.data, self.labels)
