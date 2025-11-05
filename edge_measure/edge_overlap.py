from typing import Dict, Tuple
from .base_edge_measure import BaseEdgeMeasure

class EdgeOverlap(BaseEdgeMeasure):

    def __init__(self, network):
        super().__init__(network)

    def _nx(self):
        for attr in ("_G", "G", "graph"):
            if hasattr(self.network, attr):
                return getattr(self.network, attr)
        return self.network

    def measure(self) -> Dict[Tuple, float]:
        G = self._nx()
        scores: Dict[Tuple, float] = {}
        for u, v in G.edges():
            Nu = set(G.neighbors(u)) - {v}
            Nv = set(G.neighbors(v)) - {u}
            inter = len(Nu & Nv)
            union = len(Nu | Nv)
            ov = inter / union if union > 0 else 0.0

            # store edge key in a consistent tuple order
            key = (u, v) if u <= v else (v, u)
            scores[key] = ov
        return scores

    def print_info(self) -> None:
        scores = self.measure()
        print("Edge Overlap (Highest to Lowest):")
        for (u, v), s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            print(f"Edge ({u!r}, {v!r}): {s}")
