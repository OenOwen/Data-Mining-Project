import numpy as np

class distance_measure: 
    def euclidean_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def manhattan_distance(point1, point2):
        point1, point2 = np.array(point1), np.array(point2)
        return np.sum(np.abs(point1 - point2))
    
    def circular_distance(point1, point2, period=360.0):
        point1, point2 = np.array(point1), np.array(point2)
        delta = np.abs(point1 - point2) % period
        c = np.minimum(delta, period - delta)
        return float(np.sqrt(np.sum(c ** 2)))