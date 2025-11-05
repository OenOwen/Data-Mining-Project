import numpy as np
from distance_measure import distance_measure
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

            def compute_distance_matrix(data, metric):
                D = np.zeros((data.shape[0], data.shape[0]))
                for i in range(data.shape[0]):
                    for j in range(i + 1, data.shape[0]):
                        if metric == 'euclidean':
                            d = distance_measure.euclidean_distance(data[i], data[j])
                        elif metric == 'manhattan':
                            d = distance_measure.manhattan_distance(data[i], data[j])
                        else:
                            d = np.linalg.norm(data[i] - data[j])
                        D[i, j] = D[j, i] = d
                D[D == 0] = 1e-9
                return D

            D = compute_distance_matrix(X, self.distance_measure)
            scale = np.sum(D) / 2

            rng = np.random.default_rng(self.random_state)
            pca = PCA(n_components=self.n_components, random_state=rng.integers(10000))
            Y = pca.fit_transform(X)
            Y += rng.normal(scale=1e-4, size=(n_samples, self.n_components))

            def sammon_stress(Y_flat):
                Y_reshaped = Y_flat.reshape(n_samples, self.n_components)
                D_hat = compute_distance_matrix(Y_reshaped, self.distance_measure)
                delta = (D - D_hat) / D
                return np.sum(delta ** 2) / scale

            def sammon_gradient(Y_flat):
                Y_reshaped = Y_flat.reshape(n_samples, self.n_components)
                D_hat = compute_distance_matrix(Y_reshaped, self.distance_measure)
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