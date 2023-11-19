import numpy as np
import cvxopt
from sklearn.metrics import pairwise_distances


class OneClassSVM:
    def __init__(self, kernel='rbf', gamma=0.1, nu=0.5):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.alpha = None
        self.support_vectors = None
        self.rho = None

    def _rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def _kernel_matrix(self, X):
        return pairwise_distances(X, X, metric=self._rbf_kernel)

    def _quadprog_solve_qp(self, P):
        n_samples, _ = P.shape
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(np.zeros(n_samples))
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(
            np.hstack((np.zeros(n_samples), 1 / (self.nu * n_samples) * np.ones(n_samples))))
        A = cvxopt.matrix(np.ones((1, n_samples)))
        b = cvxopt.matrix(1.0)

        # Solve the QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def fit(self, X):
        # Calculate Kernel matrix
        K = self._kernel_matrix(X)

        mu = self._quadprog_solve_qp(K)
        idx_support = np.where(np.abs(mu) > 1e-5)[0]
        mu_support = mu[idx_support]

        # Calculate rho
        index = int(np.argmin(mu_support))
        K_support = K[idx_support][:, idx_support]
        self.rho = mu_support.dot(K_support[index])
        self.support_vectors = X[idx_support]
        self.alpha = mu_support

    def score(self, X):
        decision_values = self.decision_function(X)
        score = np.mean(decision_values > 0)
        return score

    def decision_function(self, X):
        K = np.array([[self._rbf_kernel(sv, x) for x in X]
                     for sv in self.support_vectors])
        return np.dot(self.alpha, K) - self.rho

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def get_params(self, deep=True):
        # Return a dictionary of parameters
        return {"kernel": self.kernel, "gamma": self.gamma, "nu": self.nu}

    def set_params(self, **parameters):
        # Set parameters based on input dictionary
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
