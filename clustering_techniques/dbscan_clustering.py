from sklearn.cluster import DBSCAN
from clustering_techniques.base_clustering import BaseClustering
from distance_measure import distance_measure

class DBSCANClustering(BaseClustering):
    def __init__(self, dataset, eps=0.5, min_samples=5, metric='euclidean'):
        super().__init__(dataset)
        self.eps = eps
        self.min_samples = min_samples

        metric_func = {
            'euclidean': distance_measure.euclidean_distance,
            'manhattan': distance_measure.manhattan_distance,
            'circular': distance_measure.circular_distance
        }
        self.metric = metric_func.get(metric)

        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=self.metric
        )

    def fit(self):
        self.model.fit(self.dataset.getData())
        return self.model.labels_, None
