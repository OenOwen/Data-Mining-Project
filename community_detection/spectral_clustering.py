import numpy as np
from sklearn.base import defaultdict
from sklearn.cluster import KMeans
import networkx as nx

from .base_community_detection import BaseCommunityDetection

class SpectralClustering(BaseCommunityDetection):
    def community(self, max_communities=10):
        G = nx.Graph()
        G.add_nodes_from(self.network.nodes())
        G.add_edges_from(self.network.edges())
        
        n = G.number_of_nodes()
        if n <= 1:
            return [list(G.nodes())] if n == 1 else []
        
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        A = np.zeros((n, n))
        for u, v in G.edges():
            i = node_to_idx[u]; j = node_to_idx[v]
            A[i,j] = A[j,i] = 1
        
        D = np.diag(A.sum(axis=1))
        
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        
        eigvals, eigvecs = np.linalg.eigh(L)
        
        k_max = min(max_communities, n//2)
        gaps = np.diff(eigvals[1:k_max+1])
        n_communities = np.argmax(gaps) + 2
        n_communities = max(2, min(n_communities, n))
        
        X = eigvecs[:, 1:n_communities]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        
        labels = KMeans(n_clusters=n_communities, n_init=20, random_state=42).fit_predict(X)
        
        comms = defaultdict(list)
        for i, label in enumerate(labels):
            comms[label].append(nodes[i])
        
        result = [sorted(c) for c in comms.values() if c]
        result.sort(key=lambda x: x[0])
        return result

