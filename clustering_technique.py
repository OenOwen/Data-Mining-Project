from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from dataset import Dataset
from distance_measure_clustering import distance_measure_clustering

class clustering:

    def __init__(self, dataset):
        self.dataset = dataset


    def kmeans(self,  n_components=3):
        kmeans = KMeans(n_clusters=n_components)
        kmeans.fit(self.dataset.getData())
        return kmeans.labels_, kmeans.cluster_centers_
    
    def dbscan(self, eps=0.5, min_samples=5, distance_measure=distance_measure_clustering.euclidean_distance):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_measure)
        dbscan.fit(self.dataset.getData())
        return dbscan.labels_, None
    
    def agglomerative(self, n_components=3, linkage='average', distance_measure=distance_measure_clustering.euclidean_distance):
        agg = AgglomerativeClustering(n_clusters=n_components, affinity=distance_measure, linkage=linkage)
        agg.fit(self.dataset.getData())
        return agg.labels_, None