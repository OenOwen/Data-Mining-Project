from typing import Dict

from .base_node_measure import BaseNodeMeasure


class DegreeCentrality(BaseNodeMeasure):
    """
    Measures how the number of edges each node has
    """

    def __init__(self, network, normalize: bool = False):
        super().__init__(network)
        self.normalize = normalize

    def measure(self) -> Dict[int, float]:
        nodes = self.network.nodes()
        if not nodes:
            return {}

        if self.normalize:
            n = len(nodes)
            denom = max(1, n - 1)
            return {u: float(len(self.network.neighbors(u)) / denom) for u in nodes}
        else:
            return {u: float(len(self.network.neighbors(u))) for u in nodes}


    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print("Degree Centrality Measures (Highest to Lowest):")
        for node, centrality in sorted_measures:
            print(f"Node {node}: {centrality}")