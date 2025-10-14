from sklearn.cluster import AgglomerativeClustering
from .base_clustering import BaseClustering
from distance_measure.distance_measure import distance_measure

class AgglomerativeClusteringModel(BaseClustering):
    def __init__(self, dataset, n_components=3, linkage='average', metric=distance_measure.euclidean_distance):
        super().__init__(dataset)
        self.n_components = n_components
        self.linkage = linkage
        self.metric = metric
        self.model = AgglomerativeClustering(
            n_clusters=n_components,
            linkage=linkage,
            metric=metric
        )

    def fit(self):
        self.model.fit(self.dataset.getData())
        return self.model.labels_, None
