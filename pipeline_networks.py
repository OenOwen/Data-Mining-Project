import os
from network.UndirectedNetwork import UndirectedNetwork

# community detection
from community_detection.girvan_newman import GirvanNewman
from community_detection.newman_fast import NewmanFast
from community_detection.spectral_clustering import SpectralClustering

# node measure
from node_measure.degree_centrality import DegreeCentrality
from node_measure.closeness_centrality import ClosenessCentrality
from node_measure.betweenness_centrality import BetweennessCentrality

# edge measure
from edge_measure.edge_betweenness import EdgeBetweennessCentrality
from edge_measure.edge_weight import EdgeWeight
from edge_measure.edge_overlap import EdgeOverlap


def print_network_data(G):
    print("Graph loaded.")
    print(f"Nodes: {G.num_nodes()}")
    print(f"Edges: {G.num_edges()}")
    print("Nodes:", G.nodes(), "\n")
    print("Edges:", G.edges(), "\n")

def run_community_detection(G):
    
    # helper functions
    def num_clusters(communities):
        return len(communities)

    def avg_cluster_size(communities):
        n = len(communities)
        return (sum(len(c) for c in communities) / n) if n else 0.0

    # Run methods
    gn = GirvanNewman(G).community()
    nf = NewmanFast(G).community()
    sc = SpectralClustering(G).community()

    # Print concise stats
    print(f"Girvan Newman: num_clusters={num_clusters(gn):2d}, avg_size={avg_cluster_size(gn):.2f}")
    print(f"Newman Fast: num_clusters={num_clusters(nf):2d}, avg_size={avg_cluster_size(nf):.2f}")
    print(f"Spectral Clustering: num_clusters={num_clusters(sc):2d}, avg_size={avg_cluster_size(sc):.2f}")


def run_node_measures(G):
    def print_topn(title, scores, n=5):
        items = scores.items() if isinstance(scores, dict) else list(scores)
        items = [(n, float(s)) for n, s in items]
        items.sort(key=lambda x: x[1], reverse=True)
        print(f"Top {n} {title}:")
        for node, score in items[:n]:
            print(f"  {node}: {score:.6f}")
        print()

    # Degree
    dc = DegreeCentrality(G)
    dc_scores = dc.measure()    
    print_topn("Degree centrality", dc_scores)

    # Closeness
    cc = ClosenessCentrality(G)
    cc_scores = cc.measure()
    print_topn("Closeness centrality", cc_scores)

    # Betweenness
    bc = BetweennessCentrality(G)
    bc_scores = bc.measure()
    print_topn("Betweenness centrality", bc_scores)

    
    
def run_edge_measures(G):
    
    def print_topn(title, scores, n=5):
        items = scores.items() if isinstance(scores, dict) else list(scores)
        items = [(k, float(v)) for k, v in items]
        items.sort(key=lambda x: x[1], reverse=True)
        print(f"Top {n} {title}:")
        for key, score in items[:n]:
            print(f"  {key}: {score:.6f}")
        print()

    # Edge betweenness
    ebtw = EdgeBetweennessCentrality(G)
    ebtw_scores = ebtw.measure()
    print_topn("Edge betweenness", ebtw_scores)

    # Edge weight (for unweighted graphs this will all be 1.0!)
    # And we only have unweighted graphs so so 1 for all.
    ewt = EdgeWeight(G)
    ewt_scores = ewt.measure()
    print_topn("Edge weight", ewt_scores)
    
    eol = EdgeOverlap(G)
    eol_scores = eol.measure()
    print_topn("Edge Overlap", eol_scores)



def main():
        
    BASE = os.path.dirname(__file__)

    LES_MISERABLES_PATH = os.path.join(BASE, "data", "les_miserables.edgelist")
    g_less_miserables = UndirectedNetwork.from_edgelist(LES_MISERABLES_PATH)
    
    KARATE_PATH = os.path.join(BASE, "data", "karate.adjmat")
    G_karate = UndirectedNetwork.from_adjacency_matrix(KARATE_PATH)

    THREE_COMMUNITIES_PATH = os.path.join(BASE, "data", "three_communities.edgelist")
    G_three_communities = UndirectedNetwork.from_edgelist(THREE_COMMUNITIES_PATH)
    
    graphs_list = [
        g_less_miserables , G_karate, G_three_communities
    ]
    
    for G in graphs_list:
        print_network_data(G)
        run_community_detection(G)
        run_node_measures(G)
        run_edge_measures(G)
   

if __name__ == "__main__":
    main()
