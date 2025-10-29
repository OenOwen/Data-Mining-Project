from collections import defaultdict

class UndirectedNetwork:
    def __init__(self):
        self._adj = defaultdict(set)
        self._edges = set()

    @classmethod
    def from_edgelist(cls, path):
        g = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                u_raw, v_raw = parts[0], parts[1]
                try:
                    u = int(u_raw)
                    v = int(v_raw)
                except ValueError:
                    u, v = u_raw, v_raw
                g.add_edge(u, v)
        return g

    def add_edge(self, u, v):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in self._edges:
            return
        self._edges.add((a, b))
        self._adj[a].add(b)
        self._adj[b].add(a)

    def nodes(self):
        return list(self._adj.keys())

    def edges(self):
        return list(self._edges)

    def neighbors(self, u):
        return list(self._adj.get(u, ()))

    def num_nodes(self):
        return len(self._adj)

    def num_edges(self):
        return len(self._edges)

    def has_edge(self, u, v):
        a, b = (u, v) if u < v else (v, u)
        return (a, b) in self._edges
