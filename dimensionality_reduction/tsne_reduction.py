import numpy as np
from sklearn.manifold import TSNE
from .base_dimensionality_reduction import BaseDimensionalityReduction

class TSNEReduction(BaseDimensionalityReduction):

    def __init__(self, dataset, n_components=2, perplexity=10.0, random_state=42):
        super().__init__(dataset, n_components)
        self.perplexity = perplexity
        self.random_state = random_state

    def reduce(self):
        X = np.array(self.dataset.getData())
        tsne_model = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            random_state=self.random_state
        )
        
        return tsne_model.fit_transform(X)
