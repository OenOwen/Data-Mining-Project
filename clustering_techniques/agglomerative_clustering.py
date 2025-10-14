from sklearn.cluster import AgglomerativeClustering
from clustering_techniques.base_clustering import BaseClustering
from distance_measure import distance_measure

class AgglomerativeClusteringModel(BaseClustering):
    def __init__(self, dataset, n_components=3, linkage='average', metric='euclidean'):
        super().__init__(dataset)
        self.n_components = n_components
        self.linkage = linkage

        # Map string to function
        metric_func = {
            'euclidean': distance_measure.euclidean_distance,
            'manhattan': distance_measure.manhattan_distance,
            'circular': distance_measure.circular_distance
        }
        self.metric = metric_func.get(metric)

        self.model = AgglomerativeClustering(
            n_clusters=n_components,
            linkage=linkage,
            metric=self.metric
        )

    def fit(self):
        self.model.fit(self.dataset.getData())
        return self.model.labels_, None
