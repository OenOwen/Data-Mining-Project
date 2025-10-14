from sklearn.manifold import trustworthiness as sk_trustworthiness
from .base_quality_measure_dr import BaseQualityMeasureDR

class TrustworthinessMeasure(BaseQualityMeasureDR):
    def evaluate(self):
        return sk_trustworthiness(
            self.original_data,
            self.reduced_data,
            n_neighbors=self.n_neighbors
        )
