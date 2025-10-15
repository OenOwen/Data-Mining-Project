from dataset import Dataset

# Import clustering algorithms
from clustering_techniques.kmeans_clustering import KMeansClustering
from clustering_techniques.dbscan_clustering import DBSCANClustering
from clustering_techniques.agglomerative_clustering import AgglomerativeClusteringModel
from clustering_techniques.hierarchal_clustering import HierarchicalClustering


# Import clustering quality measures
from quality_measure_clustering.silhouette_measure import SilhouetteMeasure
from quality_measure_clustering.calinski_harabasz_measure import CalinskiHarabaszMeasure
from quality_measure_clustering.davies_bouldin_measure import DaviesBouldinMeasure


# --------------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------------
data = Dataset("artificial_dataset.csv")
dataset_data = data.getData()

# --------------------------------------------------------------------
# Run clustering algorithms
# --------------------------------------------------------------------
kmeans = KMeansClustering(data, n_components=3)
dbscan = DBSCANClustering(data, eps=0.5, min_samples=5)
hier_clust = HierarchicalClustering(data, n_components=3, metric='circular')
# agg = AgglomerativeClusteringModel(data, n_components=3)

# Fit and collect results
kmeans_labels, _ = kmeans.fit()
dbscan_labels, _ = dbscan.fit()
hier_clust_labels, _ = hier_clust.fit()
# agg_labels, _ = agg.fit()

cluster_results = {
    "K-Means": kmeans_labels,
    "DBSCAN": dbscan_labels,
    "Hierarchical": hier_clust_labels
}

# --------------------------------------------------------------------
# Evaluate clustering quality
# --------------------------------------------------------------------
print("=" * 70)
print("Clustering Quality Evaluation")
print("=" * 70)
print()

results_table = []

for method_name, labels in cluster_results.items():
    print(f"\n{method_name} Results:")
    print("-" * 50)

    # Initialize each quality measure
    sil = SilhouetteMeasure(data, labels)
    cal = CalinskiHarabaszMeasure(data, labels)
    dav = DaviesBouldinMeasure(data, labels)

    # Evaluate
    silhouette_score = sil.evaluate()
    calinski_score = cal.evaluate()
    davies_score = dav.evaluate()

    # Print results
    print(f"  Silhouette Score:          {silhouette_score:.4f}")
    print(f"  Calinski-Harabasz Score:   {calinski_score:.4f}")
    print(f"  Davies-Bouldin Score:      {davies_score:.4f}")

    results_table.append({
        "Method": method_name,
        "Silhouette": silhouette_score,
        "Calinski-Harabasz": calinski_score,
        "Davies-Bouldin": davies_score
    })

# --------------------------------------------------------------------
# Summary Table
# --------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'Method':<15} {'Silhouette':<20} {'Calinski-Harabasz':<25} {'Davies-Bouldin':<20}")
print("-" * 80)

for result in results_table:
    print(f"{result['Method']:<15} "
          f"{result['Silhouette']:<20.4f} "
          f"{result['Calinski-Harabasz']:<25.4f} "
          f"{result['Davies-Bouldin']:<20.4f}")

print("=" * 70)

# --------------------------------------------------------------------
# Find Best Method for Each Metric
# --------------------------------------------------------------------
print("\nBest performing methods:")
print("-" * 50)

# For Silhouette and Calinski-Harabasz, higher = better
# For Davies-Bouldin, lower = better

best_sil = max(results_table, key=lambda x: x["Silhouette"])
best_cal = max(results_table, key=lambda x: x["Calinski-Harabasz"])
best_dav = min(results_table, key=lambda x: x["Davies-Bouldin"])

print(f"  Silhouette Score:          {best_sil['Method']} ({best_sil['Silhouette']:.4f})")
print(f"  Calinski-Harabasz Score:   {best_cal['Method']} ({best_cal['Calinski-Harabasz']:.4f})")
print(f"  Davies-Bouldin Score:      {best_dav['Method']} ({best_dav['Davies-Bouldin']:.4f})")

print("=" * 70)
