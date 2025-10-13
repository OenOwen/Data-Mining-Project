from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

class quality_measure_clustering:
    def __init__(self, dataset, labels):
        # Keep a copy of the dataset and the labels produced by the clustering algorithm
        
        self.data = dataset.getData()
        self.labels = np.array(labels)

    def silhouette(self):
        # Checks how well-separated the clusters are (higher is better)
        # Range -1.0 -> 1.0

        if len(np.unique(self.labels)) < 2:
            return float('nan')
        return silhouette_score(self.data, self.labels)

    def calinski_harabasz(self):
        # Compares between-cluster variance to within-cluster variance
        # Larger values mean clusters are more distinct
        # Range 0 -> infinity
        
        if len(np.unique(self.labels)) < 2:
            return float('nan')
        return calinski_harabasz_score(self.data, self.labels)

    def davies_bouldin(self):
        # Calculates how similar clusters are to each other (lower is better)
        # Range 0 -> infinity

        if len(np.unique(self.labels)) < 2:
            return float('nan')
        return davies_bouldin_score(self.data, self.labels)
