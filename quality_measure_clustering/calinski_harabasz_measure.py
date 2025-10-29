from sklearn.metrics import calinski_harabasz_score
from .base_quality_measure import BaseQualityMeasure

class CalinskiHarabaszMeasure(BaseQualityMeasure):

    def evaluate(self):
        if not self._valid_labels():
            return float('nan')
        return calinski_harabasz_score(self.data, self.labels)
