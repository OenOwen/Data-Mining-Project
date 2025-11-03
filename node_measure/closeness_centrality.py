from collections import deque
from typing import Dict
from networkx import closeness_centrality

from .base_node_measure import BaseNodeMeasure


class ClosenessCentrality(BaseNodeMeasure):
    """
    Measures how close a node is to all other nodes, when talking about shortest paths.
    If a node has high closeness, it can quickly reach others
    """

    def __init__(self, network):
        super().__init__(network)

    def measure(self) -> Dict[int, float]:
        clos = closeness_centrality(self.network.to_networkx())
        return clos
        

    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print("Closeness Centrality Measures (Highest to Lowest):")
        for node, centrality in sorted_measures:
            print(f"Node {node}: {centrality}")