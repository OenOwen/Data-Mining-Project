import networkx as nx
from .base_community_detection import BaseCommunityDetection
from networkx.algorithms.community import greedy_modularity_communities


class NewmanFast(BaseCommunityDetection):
    def __init__(self, network, num_communities=2, resolution: float = 1.0):
        super().__init__(network)
        self.num_communities = num_communities
        self.resolution = resolution

    def community(self):
        G = nx.Graph()
        G.add_nodes_from(self.network.nodes())
        G.add_edges_from(self.network.edges())

        if G.number_of_nodes() == 0:
            return []
        if G.number_of_edges() == 0:
            return [[node] for node in sorted(G.nodes())]

        kwargs = {"resolution": self.resolution}
        kwargs["best_n"] = int(self.num_communities)
        communities = greedy_modularity_communities(G, **kwargs)


        result = [sorted(community) for community in communities]

        if len(result) < int(self.num_communities):
            target = min(int(self.num_communities), G.number_of_nodes())
            comms = [set(c) for c in result]
            while len(comms) < target:
                comms.sort(key=len, reverse=True)
                C = comms.pop(0)
                if len(C) <= 1:
                    comms.append(C)
                    break

                H = G.subgraph(C)
                deg = dict(H.degree())
                ordered = sorted(C, key=lambda u: deg.get(u, 0), reverse=True)
                mid = max(1, len(ordered) // 2)
                A, B = set(ordered[:mid]), set(ordered[mid:])
                
                if not A or not B:
                    node = next(iter(C))
                    A, B = {node}, C - {node}
                comms.extend([A, B])
            result = [sorted(c) for c in comms]

        result.sort(key=lambda x: x[0])
        self.communities = result
        return result
