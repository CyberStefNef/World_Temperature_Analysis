import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100, n_init=10):
        self.k = k
        self.max_iter = max_iter
        # Number of times the KMeans algorithm will be run with different centroid seeds
        self.n_init = n_init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def __init_centroids(self, data):
        indices = np.random.choice(len(data), size=self.k, replace=False)
        return data[indices]

    def __compute_inertia(self, data, centroids, labels):
        return np.sum(np.linalg.norm(data - centroids[labels], axis=1)**2)

    def __compute_distances(self, data, centroids):
        """Compute distances between each data point and the centroids."""
        return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

    def fit(self, data):
        data = np.array(data, dtype=float)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self.__init_centroids(data)
            for _ in range(self.max_iter):
                distances = self.__compute_distances(data, centroids)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array(
                    [data[labels == i].mean(axis=0) for i in range(self.k)])
                if np.all(new_centroids == centroids):
                    break
                centroids = new_centroids
            inertia = self.__compute_inertia(data, centroids, labels)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.inertia_ = best_inertia
        self.centroids = best_centroids
        self.labels_ = best_labels

    def predict(self, data):
        data = np.array(data, dtype=float)
        distances = self.__compute_distances(data, self.centroids)
        return np.argmin(distances, axis=1)
