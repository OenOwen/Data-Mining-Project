import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from .base_dimensionality_reduction import BaseDimensionalityReduction


class SammonMappingReduction(BaseDimensionalityReduction):
    def __init__(self, dataset, n_components=2, max_iter=300, tol=1e-9, random_state=None):
        super().__init__(dataset, n_components)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def reduce(self):
        X = np.asarray(self.dataset.getData(), dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, _ = X.shape

        D = squareform(pdist(X))
        D[D == 0] = 1e-9

        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(scale=1e-4, size=(n_samples, self.n_components))

        scale = (D.sum() / 2)

        def sammon_stress(Y_flat):
            Y_reshaped = Y_flat.reshape(n_samples, self.n_components)
            D_hat = squareform(pdist(Y_reshaped))
            D_hat[D_hat == 0] = 1e-9
            delta = (D - D_hat)
            stress = (delta**2 / D).sum() / scale
            return stress

        res = minimize(
            sammon_stress,
            Y.ravel(),
            method="L-BFGS-B"
        )

        Y_opt = res.x.reshape(n_samples, self.n_components)

        self.reduced_data = Y_opt
        return Y_opt
