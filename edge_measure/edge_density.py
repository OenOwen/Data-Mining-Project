from typing import Dict, Tuple
from .base_edge_measure import BaseEdgeMeasure


class EdgeDensity(BaseEdgeMeasure):

    def __init__(self, network):
        super().__init__(network)

    def measure(self) -> Dict[str, float]:
        n = self.network.num_nodes()
        m = self.network.num_edges()
        if n <= 1:
            density = 0.0
        else:
            density = (2.0 * m) / (n * (n - 1))
        return {"density": density}

    def print_info(self) -> None:
        result = self.measure()
        print(f"Edge Density of Network: {result['density']:.4f}")
