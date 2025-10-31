import os
from network.UndirectedNetwork import UndirectedNetwork

from community_detection.girvan_newman import GirvanNewman
from community_detection.newman_fast import NewmanFast
from community_detection.spectral_clustering import SpectralClustering

from node_measure.degree_centrality import DegreeCentrality
from node_measure.closeness_centrality import ClosenessCentrality
from node_measure.betweenness_centrality import BetweennessCentrality

# edge measure
from edge_measure.edge_betweenness import EdgeBetweennessCentrality
from edge_measure.edge_weight import EdgeWeight
from edge_measure.edge_density import EdgeDensity

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "les_miserables.edgelist")

def main():
    
    # Create new undirected nw from datapath
    G = UndirectedNetwork.from_edgelist(DATA_PATH)
    
    print("Graph loaded.")
    print(f"Nodes: {G.num_nodes()}")
    print(f"Edges: {G.num_edges()}")
    print("Nodes:", G.nodes())
    print("Edges:", G.edges())

    print(f"GirvanNewman Community: {GirvanNewman(G).community()}")
    print(f"NewmanFast Community: {NewmanFast(G).community()}")
    print(f"SpectralClustering Community: {SpectralClustering(G).community()}")

    dc = DegreeCentrality(G)
    dc.measure()
    dc.print_info()

    cc = ClosenessCentrality(G)
    cc.measure()
    cc.print_info()

    bc = BetweennessCentrality(G)
    bc.measure()
    bc.print_info()
    
    
    # EDGE MEASURE
    
    # edge bewteenness
    ebtw = EdgeBetweennessCentrality(G)
    ebtw.print_info()
    
    
    # edge weight
    ewt = EdgeWeight(G)
    ewt.print_info()
    
    # edge density
    ed = EdgeDensity(G)
    ed.print_info()

if __name__ == "__main__":
    main()
