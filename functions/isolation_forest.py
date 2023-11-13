import numpy as np

class IsolationForest:
    def __init__(self, n_trees=100, subsample_size=256):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = []

    def forest(self, data):
        for i in range(self.n_trees):
            subsample = data[np.random.choice(data.shape[0], self.subsample_size, replace=False)]
            itree = self.isolation_tree(subsample)
            self.trees.append(itree)

    def isolation_forest_anomaly_score(self, data, tree):
        scores = np.array([self.path_length(tree, point) for point in data])
        return 2 ** (-scores / len(self.trees))

    def isolation_tree(self, data, height=0, max_height=None):
        if max_height is None:
            max_height = int(np.log2(len(data)))

        if max_height <= 0 or len(data) <= 1:
            return {
                "data": data,
                "height": height
            }
        else:
            split_attribute = np.random.randint(0, data.shape[1])
            split_value = np.random.uniform(data[:, split_attribute].min(), data[:, split_attribute].max())
            left_data = data[data[:, split_attribute] < split_value]
            right_data = data[data[:, split_attribute] >= split_value]
            return {
                "split_attribute": split_attribute,
                "split_value": split_value,
                "left": self.isolation_tree(left_data, height + 1, max_height),
                "right": self.isolation_tree(right_data, height + 1, max_height)
            }

    def path_length(self, tree, point, current_height=0):
        if "split_attribute" not in tree:
            return current_height + self.anomaly_score(tree)
        else:
            split_attribute = tree["split_attribute"]
            if point[split_attribute] < tree["split_value"]:
                return self.path_length(tree["left"], point, current_height + 1)
            else:
                return self.path_length(tree["right"], point, current_height + 1)

    def anomaly_score(self, tree):
        if self.subsample_size == 2:
            return 1
        elif self.subsample_size < 2:
            return 0
        else:
            return 2 * (np.log(2 ** tree["height"] - 1) + np.euler_gamma) - (2 * (tree["height"] - 1) / (2 ** tree["height"] - 1))
            
    