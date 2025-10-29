from sklearn.neighbors import NearestNeighbors
from .base_quality_measure_dr import BaseQualityMeasureDR

class NeighborhoodPreservationMeasure(BaseQualityMeasureDR):
    def evaluate(self):
        knn_orig = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.original_data)
        knn_reduced = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.reduced_data)

        _, neighbors_orig = knn_orig.kneighbors(self.original_data)
        _, neighbors_reduced = knn_reduced.kneighbors(self.reduced_data)

        n_samples = self.original_data.shape[0]
        preserved = 0
        for i in range(n_samples):
            preserved += len(set(neighbors_orig[i]) & set(neighbors_reduced[i])) / self.n_neighbors

        return preserved / n_samples
