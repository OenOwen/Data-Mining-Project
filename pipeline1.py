from dataset import Dataset

from distance_measure import distance_measure

from clustering_techniques.dbscan_clustering import DBSCANClustering
from clustering_techniques.kmeans_clustering import KMeansClustering
from clustering_techniques.hierarchal_clustering import HierarchicalClustering

from quality_measure_clustering.calinski_harabasz_measure import CalinskiHarabaszMeasure
from quality_measure_clustering.davies_bouldin_measure import DaviesBouldinMeasure
from quality_measure_clustering.silhouette_measure import SilhouetteMeasure

artificial_data = Dataset("data/artificial_dataset.csv")
wine_data = Dataset("data/winequality-red.csv")
magic_data = Dataset("data/magic04.csv")

wine_data = wine_data.reduceData(1000)
magic_data = magic_data.reduceData(1000)

datasets = {
    "Artificial": artificial_data,
    "Wine Quality": wine_data,
    "Magic04": magic_data
}

quality_measure_classes = {
    "Calinski-Harabasz": CalinskiHarabaszMeasure,
    "Davies-Bouldin": DaviesBouldinMeasure,
    "Silhouette": SilhouetteMeasure
}

cluster_list = {
    dataset_name: {
        "KMeans": KMeansClustering(dataset),
        "DBSCAN": DBSCANClustering(dataset, min_samples=2),
        "Hierarchical": HierarchicalClustering(dataset)
    }
    for dataset_name, dataset in datasets.items()
}


for dataset_name, clustering_algorithms in cluster_list.items():
    dataset = datasets[dataset_name]
    labels = None
    print(f"\nDataset: {dataset_name}")
    print("=" * 50)

    for cluster_name, clustering_algorithm in clustering_algorithms.items():
        print(f"\nClustering Algorithm: {cluster_name}")
        labels, centers = clustering_algorithm.fit()

        for measure_name, measure_cls in quality_measure_classes.items():
            quality_measure = measure_cls(dataset, labels)
            score = quality_measure.evaluate()
            print(f"  {measure_name} Score: {score:.4f}")


