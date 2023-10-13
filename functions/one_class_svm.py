import numpy as np
import cvxopt


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
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._rbf_kernel(X[i], X[j])
        return K

    def fit(self, X):
        n_samples, n_features = X.shape

        # Calculate Kernel matrix
        K = self._kernel_matrix(X)

        # Setup the quadratic programming problem
        P = cvxopt.matrix(K)
        q = cvxopt.matrix(-np.ones((n_samples, 1)))
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(
            np.hstack((np.zeros(n_samples), self.nu * n_samples * np.ones(n_samples))))
        A = cvxopt.matrix(np.ones((1, n_samples)))
        b = cvxopt.matrix(1.0)

        # Solve the QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alphas = np.ravel(solution['x'])
        idx = (alphas > 1e-6)
        self.alpha = alphas[idx]
        self.support_vectors = X[idx]

        # Calculate rho
        self.rho = np.mean(
            [np.sum(self.alpha * self._rbf_kernel(self.support_vectors, x))
             for x in self.support_vectors]
        )

    def decision_function(self, X):
        return np.array([
            np.sum(self.alpha * self._rbf_kernel(self.support_vectors, x)) - self.rho for x in X
        ])

    def predict(self, X):
        return np.sign(self.decision_function(X))
