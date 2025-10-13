from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class dimensionality_reduction:
    def __init__(self, dataset, n_components=2):
        self.dataset = dataset
        self.n_components = n_components

    def pca(self):
        pca_model = PCA(n_components=self.n_components)
        return pca_model.fit_transform(self.dataset.getData())

    # TODO t-SNE not working
    def tsne(self):
        tsne_model = TSNE(n_components=self.n_components, random_state=42)
        return tsne_model.fit_transform(self.dataset.getData())

    def umap(self):
        umap_model = UMAP(n_components=self.n_components, random_state=42)
        return umap_model.fit_transform(self.dataset.getData())
