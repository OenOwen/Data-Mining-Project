from dataset import Dataset

import time

from clustering_techniques.dbscan_clustering import DBSCANClustering
from clustering_techniques.kmeans_clustering import KMeansClustering
from clustering_techniques.hierarchal_clustering import HierarchicalClustering

from quality_measure_clustering.calinski_harabasz_measure import CalinskiHarabaszMeasure
from quality_measure_clustering.davies_bouldin_measure import DaviesBouldinMeasure
from quality_measure_clustering.silhouette_measure import SilhouetteMeasure

from dimensionality_reduction.pca_reduction import PCAReduction
from dimensionality_reduction.sammon_mapping_reduction import SammonMappingReduction
from dimensionality_reduction.tsne_reduction import TSNEReduction

from quality_measure_dimensionality_reduction.distance_correlation_measure import DistanceCorrelationMeasure
from quality_measure_dimensionality_reduction.neighborhood_preservation_measure import NeighborhoodPreservationMeasure
from quality_measure_dimensionality_reduction.trustworthiness_measure import TrustworthinessMeasure

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
        "DBSCAN": DBSCANClustering(dataset, min_samples=2, metric='euclidean'),
        "Hierarchical": HierarchicalClustering(dataset, metric='euclidean')
    }
    for dataset_name, dataset in datasets.items()
}

time_list_clustering = {}

quality_measure_results = {}

for dataset_name, clustering_algorithms in cluster_list.items():
    dataset = datasets[dataset_name]
    labels = None
    print(f"\nDataset: {dataset_name}")
    print("=" * 50)

    for cluster_name, clustering_algorithm in clustering_algorithms.items():
        print(f"\nClustering Algorithm: {cluster_name}")
        start_time = time.time()
        labels, centers = clustering_algorithm.fit()
        end_time = time.time()
        time_list_clustering[(dataset_name, cluster_name)] = end_time - start_time

        for measure_name, measure_cls in quality_measure_classes.items():
            quality_measure = measure_cls(dataset, labels)
            score = quality_measure.evaluate()
            quality_measure_results[(dataset_name, cluster_name, measure_name)] = score
            print(f"  {measure_name} Score: {score:.4f}")



# Used for Requirements 2
time_list_dr = {}

quality_measure_results_dr = {}

reductions = [PCAReduction, SammonMappingReduction, TSNEReduction]
measures = [DistanceCorrelationMeasure, NeighborhoodPreservationMeasure, TrustworthinessMeasure]

for data_name in datasets:
    dataset = datasets[data_name]
    for reduction in reductions:
        reduced_data = reduction(dataset).reduce()
        for measure in measures:
            quality = measure(dataset.getData(), reduced_data).evaluate()


for dataset_name, clustering_algorithms in cluster_list.items():
    dataset = datasets[dataset_name]
    labels = None

    for cluster_name, clustering_algorithm in clustering_algorithms.items():
        start_time = time.time()
        labels, centers = clustering_algorithm.fit()
        end_time = time.time()
        time_list_dr[(dataset_name, cluster_name)] = end_time - start_time

        for measure_name, measure_cls in quality_measure_classes.items():
            quality_measure = measure_cls(dataset, labels)
            score = quality_measure.evaluate()
            quality_measure_results_dr[(dataset_name, cluster_name, measure_name)] = score


print("\nClustering Times:")
for (dataset_name, cluster_name), elapsed_time in time_list_clustering.items():
    print(f"Dataset: {dataset_name}, Algorithm: {cluster_name}, Time: {elapsed_time:.4f} seconds")
print("\nDimensionality Reduction Times:")
for (dataset_name, cluster_name), elapsed_time in time_list_dr.items():
    print(f"Dataset: {dataset_name}, Algorithm: {cluster_name}, Time: {elapsed_time:.4f} seconds")

print("\nClustering Quality Measures:")
for (dataset_name, cluster_name, measure_name), score in quality_measure_results.items():
    print(f"Dataset: {dataset_name}, Algorithm: {cluster_name}, Measure: {measure_name}, Score: {score:.4f}")
print("\nDimensionality Reduction Quality Measures:")
for (dataset_name, cluster_name, measure_name), score in quality_measure_results_dr.items():
    print(f"Dataset: {dataset_name}, Algorithm: {cluster_name}, Measure: {measure_name}, Score: {score:.4f}")