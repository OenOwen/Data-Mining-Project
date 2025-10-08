import distance_measure
from dataset import Dataset
from clustering_technique import clustering
from sklearn.cluster import DBSCAN

data = Dataset("artificial_dataset.csv")

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan.fit(data.getData())
print("DBSCAN Labels:", dbscan.labels_)

clust = clustering(data)
labels, centers = clust.dbscan(eps=0.5, min_samples=5)

print("Labels:", labels)
print("Centers:", centers)

