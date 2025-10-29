import os
from network.UndirectedNetwork import UndirectedNetwork

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "karate.edgelist")

def main():
    
    # Create new undirected nw from datapath
    G = UndirectedNetwork.from_edgelist(DATA_PATH)
    
    print("Graph loaded.")
    print(f"Nodes: {G.num_nodes()}")
    print(f"Edges: {G.num_edges()}")
    print("Nodes:", G.nodes())
    print("Edges:", G.edges())

if __name__ == "__main__":
    main()
