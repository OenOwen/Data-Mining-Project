import numpy as np

class distance_measure_clustering:
    
    def euclidean_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def manhattan_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.sum(np.abs(point1 - point2))

    def circular_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.min(np.abs(point1 - point2)), 360 - np.abs(point1 - point2)

    
