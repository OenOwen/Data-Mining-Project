from umap import UMAP
from .base_dimensionality_reduction import BaseDimensionalityReduction

class UMAPReduction(BaseDimensionalityReduction):
    def __init__(self, dataset, n_components=2, random_state=42):
        super().__init__(dataset, n_components)
        self.random_state = random_state

    def reduce(self):
        umap_model = UMAP(
            n_components=self.n_components,
            random_state=self.random_state
        )
        
        return umap_model.fit_transform(self.dataset.getData())
