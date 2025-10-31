from typing import Dict, Tuple
from .base_edge_measure import BaseEdgeMeasure

class EdgeWeight(BaseEdgeMeasure):

    def __init__(self, network, default_weight: float = 1.0):
        super().__init__(network)
        self.default_weight = default_weight

    def measure(self) -> Dict[Tuple[int, int], float]:
        if hasattr(self.network, "_weights"):
            weights = {}
            for (u, v) in self.network.edges():
                key = (u, v) if u < v else (v, u)
                weight = self.network._weights.get(key, self.default_weight)
                weights[key] = float(weight)
            return weights

        return {edge: float(self.default_weight) for edge in self.network.edges()}
