import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from .base_dimensionality_reduction import BaseDimensionalityReduction

class SammonMappingReduction(BaseDimensionalityReduction):
    def __init__(self, dataset, n_components=2, max_iter=300, random_state=None):
        super().__init__(dataset, n_components)
        self.max_iter = max_iter
        self.random_state = random_state
        self.reduced_data = None

    def reduce(self):
        X = np.asarray(self.dataset.getData(), dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, _ = X.shape

        D = squareform(pdist(X))
        D[D == 0] = 1e-9
        scale = np.sum(D) / 2

        rng = np.random.default_rng(self.random_state)
        pca = PCA(n_components=self.n_components, random_state=rng.integers(10000))
        Y = pca.fit_transform(X)
        Y += rng.normal(scale=1e-4, size=(n_samples, self.n_components))

        def sammon_stress(Y_flat):
            Y_reshaped = Y_flat.reshape(n_samples, self.n_components)
            D_hat = squareform(pdist(Y_reshaped))
            D_hat[D_hat == 0] = 1e-9
            delta = (D - D_hat) / D
            return np.sum(delta ** 2) / scale

        def sammon_gradient(Y_flat):
            Y_reshaped = Y_flat.reshape(n_samples, self.n_components)
            D_hat = squareform(pdist(Y_reshaped))
            D_hat[D_hat == 0] = 1e-9
            grad = np.zeros_like(Y_reshaped)
            for i in range(n_samples):
                delta = Y_reshaped[i] - Y_reshaped
                dist = D_hat[i]
                term1 = (D[i] - dist) / (D[i] * dist + 1e-9)
                term2 = delta / (dist[:, None] + 1e-9)
                grad[i] = np.sum(term1[:, None] * term2, axis=0)
            grad *= -2 / scale
            return grad.ravel()

        res = minimize(
            sammon_stress,
            Y.ravel(),
            method="L-BFGS-B",
            jac=sammon_gradient,
            options={"maxiter": self.max_iter}
        )

        self.reduced_data = res.x.reshape(n_samples, self.n_components)
        return self.reduced_data