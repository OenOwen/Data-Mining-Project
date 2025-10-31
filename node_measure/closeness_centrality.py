from collections import deque
from typing import Dict

from .base_node_measure import BaseNodeMeasure


class ClosenessCentrality(BaseNodeMeasure):
    """
    Measures how close a node is to all other nodes, when talking about shortest paths.
    If a node has high closeness, it can quickly reach others
    """

    def __init__(self, network, normalize: bool = False):
        super().__init__(network)
        self.normalize = normalize

    def _bfs_lengths(self, source: int) -> Dict[int, int]:
        dist: Dict[int, int] = {source: 0}
        q: deque[int] = deque([source])
        while q:
            u = q.popleft()
            for v in self.network.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def measure(self) -> Dict[int, float]:
        nodes = self.network.nodes()
        n = len(nodes)
        if n == 0:
            return {}

        results: Dict[int, float] = {}
        denom_n = max(1, n - 1)

        for u in nodes:
            dist = self._bfs_lengths(u)
            s = max(0, len(dist) - 1)
            if s == 0:
                results[u] = 0.0
                continue
            totsp = sum(d for v, d in dist.items() if v != u)
            if totsp <= 0:
                results[u] = 0.0
                continue

            c = s / totsp
            if self.normalize:
                c *= (s / denom_n)
            results[u] = float(c)

        return results

    def print_info(self) -> None:
        measures = self.measure()
        sorted_measures = sorted(measures.items(), key=lambda item: item[1], reverse=True)
        print("Closeness Centrality Measures (Highest to Lowest):")
        for node, centrality in sorted_measures:
            print(f"Node {node}: {centrality}")