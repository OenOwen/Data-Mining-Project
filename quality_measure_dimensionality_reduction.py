import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

class quality_measure_dimensionality_reduction:
    def __init__(self, original_data, reduced_data, n_neighbors=5):
        self.original_data = np.array(original_data)
        self.reduced_data = np.array(reduced_data)
        self.n_neighbors = n_neighbors

    def trustworthiness(self):
        return sk_trustworthiness(
            self.original_data,
            self.reduced_data,
            n_neighbors=self.n_neighbors
        )

    def distance_correlation(self):
        dist_original = pdist(self.original_data)
        dist_reduced = pdist(self.reduced_data)
        
        return np.corrcoef(dist_original, dist_reduced)[0, 1]

    def neighborhood_preservation(self):
        knn_orig = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.original_data)
        knn_reduced = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.reduced_data)

        _, neighbors_orig = knn_orig.kneighbors(self.original_data)
        _, neighbors_reduced = knn_reduced.kneighbors(self.reduced_data)

        n_samples = self.original_data.shape[0]
        preserved = 0
        for i in range(n_samples):
            preserved += len(set(neighbors_orig[i]) & set(neighbors_reduced[i])) / self.n_neighbors

        return preserved / n_samples