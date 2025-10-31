# edge_measure/edgre_betweenness.py

from collections import deque, defaultdict
from typing import Dict, Tuple

from .base_edge_measure import BaseEdgeMeasure


class EdgeBetweennessCentrality(BaseEdgeMeasure):
   
    def __init__(self, network, normalize: bool = False):
        super().__init__(network)
        self.normalize = normalize

    def measure(self) -> Dict[Tuple[int, int], float]:
        betweenness = defaultdict(float)
        nodes = self.network.nodes()

        for s in nodes:
            stack = []
            pred = defaultdict(list)
            sigma = dict.fromkeys(nodes, 0.0)
            dist = dict.fromkeys(nodes, -1)

            sigma[s] = 1.0
            dist[s] = 0
            q = deque([s])

            while q:
                v = q.popleft()
                stack.append(v)
                for w in self.network.neighbors(v):
                    if dist[w] < 0:
                        q.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = dict.fromkeys(nodes, 0.0)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] != 0:
                        c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                        edge = tuple(sorted((v, w)))
                        betweenness[edge] += c
                        delta[v] += c

        for e in betweenness:
            betweenness[e] /= 2.0

        if self.normalize:
            n = len(nodes)
            if n > 2:
                norm = 1 / ((n - 1) * (n - 2) / 2)
                for e in betweenness:
                    betweenness[e] *= norm

        return dict(betweenness)
