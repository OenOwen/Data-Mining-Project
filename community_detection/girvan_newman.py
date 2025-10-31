import networkx as nx
from .base_community_detection import BaseCommunityDetection
from networkx.algorithms.community import girvan_newman, modularity

class GirvanNewman(BaseCommunityDetection):
    def community(self):
        G = nx.Graph()
        G.add_nodes_from(self.network.nodes())
        G.add_edges_from(self.network.edges())
        
        if G.number_of_edges() == 0:
            return [set(G.nodes())] if G.number_of_nodes() > 0 else []
        
        communities_generator = girvan_newman(G)
        
        max_communities = None
        max_mod = -1
        for communities in communities_generator:
            current_mod = modularity(G, communities)
            if current_mod > max_mod:
                max_mod = current_mod
                max_communities = communities
        
        if max_communities is None:
            return []
        return [sorted(community) for community in sorted(max_communities, key=min)]
