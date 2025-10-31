import networkx as nx
from .base_community_detection import BaseCommunityDetection
from networkx.algorithms.community import greedy_modularity_communities

class NewmanFast(BaseCommunityDetection):
    def community(self):
        G = nx.Graph()
        G.add_nodes_from(self.network.nodes())
        G.add_edges_from(self.network.edges())
        
        if G.number_of_nodes() == 0:
            return []
        if G.number_of_edges() == 0:
            return [[node] for node in sorted(G.nodes())]
        
        communities = greedy_modularity_communities(G)
        
        result = [sorted(community) for community in communities]
        result.sort(key=lambda x: x[0])
        
        return result
