import distance_measure
from dataset import Dataset
from clustering_technique import clustering
from sklearn.cluster import DBSCAN
from quality_measure import QualityMeasure


data = Dataset("artificial_dataset.csv")

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan.fit(data.getData())
print("DBSCAN Labels:", dbscan.labels_)

clust = clustering(data)
labels, centers = clust.kmeans(n_components=3)

print("Labels:", labels)
print("Centers:", centers)

qm = QualityMeasure(data, labels)

sil = qm.silhouette()
ch = qm.calinski_harabasz()
db = qm.davies_bouldin()

print("\n=== Quality Measures ===")
print(f"Silhouette Score: {sil:.4f}")
print(f"Calinski-Harabasz Index: {ch:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")