from sklearn.cluster import KMeans
from clustering_techniques.base_clustering import BaseClustering

class KMeansClustering(BaseClustering):
    def __init__(self, dataset, n_components=3):
        super().__init__(dataset)
        self.n_components = n_components
        self.model = KMeans(n_clusters=n_components)

    def fit(self):
        self.model.fit(self.dataset.getData())
        return self.model.labels_, self.model.cluster_centers_
