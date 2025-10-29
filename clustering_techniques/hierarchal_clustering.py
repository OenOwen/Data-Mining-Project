from clustering_techniques.base_clustering import BaseClustering
from distance_measure import distance_measure
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

class HierarchicalClustering(BaseClustering):

    def __init__(self, dataset, n_components=3, linkage='average', metric='euclidean'):
        super().__init__(dataset)
        self.n_components = int(n_components)
        self.linkage = linkage

        metric_func = {
            'euclidean': distance_measure.euclidean_distance,
            'manhattan': distance_measure.manhattan_distance,
            'circular': distance_measure.circular_distance
        }

        self.metric = metric_func[metric]

    def fit(self):
        X = self.dataset.getData()

        condensed = pdist(X, metric=self.metric)
        Z = linkage(condensed, method=self.linkage)

        labels = fcluster(Z, t=self.n_components, criterion='maxclust')
        labels = labels - 1

        return labels, None