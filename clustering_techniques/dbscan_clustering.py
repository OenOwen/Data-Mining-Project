from sklearn.cluster import DBSCAN
from .base_clustering import BaseClustering
from distance_measure.distance_measure import distance_measure

class DBSCANClustering(BaseClustering):
    def __init__(self, dataset, eps=0.5, min_samples=5, metric=distance_measure.euclidean_distance):
        super().__init__(dataset)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit(self):
        self.model.fit(self.dataset.getData())
        return self.model.labels_, None
