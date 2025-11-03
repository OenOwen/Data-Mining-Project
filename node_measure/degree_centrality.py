from typing import Dict
from networkx import degree_centrality

from .base_node_measure import BaseNodeMeasure


class DegreeCentrality(BaseNodeMeasure):
    """
    Measures how the number of edges each node has
    """

    def __init__(self, network):
        super().__init__(network)

    def measure(self) -> Dict[int, float]:
        deg = degree_centrality(self.network.to_networkx())
        return deg



    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print("Degree Centrality Measures (Highest to Lowest):")
        for node, centrality in sorted_measures:
            print(f"Node {node}: {centrality}")