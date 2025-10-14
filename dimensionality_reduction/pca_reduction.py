from sklearn.decomposition import PCA
from .base_dimensionality_reduction import BaseDimensionalityReduction

class PCAReduction(BaseDimensionalityReduction):

    def reduce(self):
        pca_model = PCA(n_components=self.n_components)
        return pca_model.fit_transform(self.dataset.getData())
