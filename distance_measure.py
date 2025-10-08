

class distance_measure:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def euclidean_distance(self):
        return ((self.point1[0] - self.point2[0]) ** 2 + (self.point1[1] - self.point2[1]) ** 2) ** 0.5