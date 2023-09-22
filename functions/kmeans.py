import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.__k = k
        self.__max_iter = max_iter
        self.centroids = []

    def __init_centroids(self, data):
        indices = np.random.choice(len(data), size=self.__k, replace=False)
        self.centroids = np.take(data, indices)

    def fit(self, data):
        data = np.array(data, dtype=float)
        self.__init_centroids(data)
        last_centroids = []
        for _ in range(self.__max_iter):
            categories = {}
            for point in data:
                index = np.argmin(np.absolute(self.centroids - point))
                if index not in categories:
                    categories[index] = []
                categories[index].append(point)

            for key, category in categories.items():
                self.centroids[key] = np.mean(category)

            if np.array_equal(last_centroids, self.centroids):
                return categories

            last_centroids = self.centroids

    def predict(self, data):
        pass
