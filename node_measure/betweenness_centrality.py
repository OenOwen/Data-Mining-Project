from collections import deque
from typing import Dict, List
from networkx import betweenness_centrality

from .base_node_measure import BaseNodeMeasure


class BetweennessCentrality(BaseNodeMeasure):
    """
    Measures how often a node lies on the shortest paths between pairs of other nodes.
    If a node has high betweenness, it often acts as a bridge between clusters.
    """

    def __init__(self, network, normalize: bool = False):
        super().__init__(network)
        self.normalize = normalize

    def measure(self) -> Dict[int, float]:
        betw = betweenness_centrality(self.network.to_networkx(), normalized=self.normalize)
        return betw

    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print("Betweenness Centrality Measures (Highest to Lowest):")
        for node, centrality in sorted_measures:
            print(f"Node {node}: {centrality}")
