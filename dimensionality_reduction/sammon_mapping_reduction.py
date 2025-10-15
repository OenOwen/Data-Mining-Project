from .base_dimensionality_reduction import BaseDimensionalityReduction

class SammonMappingReduction(BaseDimensionalityReduction):
    def __init__(self, dataset, n_components=2):
        super().__init__(dataset, n_components)

    def reduce(self):
        pass
