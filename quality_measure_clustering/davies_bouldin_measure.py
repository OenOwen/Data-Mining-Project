from sklearn.metrics import davies_bouldin_score
from .base_quality_measure import BaseQualityMeasure

class DaviesBouldinMeasure(BaseQualityMeasure):
    def evaluate(self):
        if not self._valid_labels():
            return float('nan')
        return davies_bouldin_score(self.data, self.labels)
