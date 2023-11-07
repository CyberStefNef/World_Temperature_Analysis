import numpy as np

# Function to construct an isolation tree
def isolation_tree(data, height=0, max_height=None):
    """
    Construct an isolation tree for the given data.

    An isolation tree is a binary tree used in isolation forest, an anomaly detection
    algorithm. The tree is built by recursively partitioning the data into two
    sub-samples until a predefined maximum height is reached or there is only one
    data point left.

    Parameters:
    -----------
    data : numpy.ndarray
        The input data to be used for constructing the isolation tree.

    height : int, optional
        The current height of the tree. This parameter is used internally for recursion
        and should not be set manually. By default, it is set to 0.

    max_height : int, optional
        The maximum height of the isolation tree. Once the tree reaches this height or
        there is only one data point left, a leaf node is created. If not specified,
        it is set to log2(len(data)), effectively limiting the tree's depth based on
        the data size.

    Returns:
    --------
    dict
        A dictionary representing a node in the isolation tree. The structure of the
        dictionary varies depending on whether it's a non-leaf or leaf node. Non-leaf
        nodes include information about the splitting attribute, split value, and
        references to their left and right child nodes. Leaf nodes contain the data
        points and their height in the tree.
    """
    if max_height is None:
        # Using a base-2 logarithm allows us to control the depth of the tree in a way that aligns with the binary tree structure.
        max_height = int(np.log2(len(data)))

    # If the maximum height is reached or there's only one data point, create a leaf node
    if max_height <= 0 or len(data) <= 1:
        return {
            "data": data,
            "height": height
        }
    else:
        # Randomly select an attribute and split value
        split_attribute = np.random.randint(0, data.shape[1])
        split_value = np.random.uniform(data[:, split_attribute].min(), data[:, split_attribute].max())
        
        # Partition data into left and right sub-samples and create non-leaf nodes
        left_data = data[data[:, split_attribute] < split_value]
        right_data = data[data[:, split_attribute] >= split_value]
        
        return {
            "split_attribute": split_attribute,
            "split_value": split_value,
            "left": isolation_tree(left_data, height + 1, max_height),
            "right": isolation_tree(right_data, height + 1, max_height)
        }
        
# Function to build an isolation forest
def isolation_forest(data, n_trees=100, subsample_size=256):
    """
    Build an Isolation Forest for anomaly detection.

    Isolation Forest is an ensemble-based anomaly detection algorithm that constructs
    a forest of isolation trees. It measures the ease with which a data point can be
    separated from the rest of the data, with anomalies being isolated more quickly.

    Parameters:
    -----------
    data : numpy.ndarray
        The input data to be used for constructing the Isolation Forest. It should be
        a two-dimensional numpy array where rows represent data points and columns
        represent features.

    n_trees : int, optional
        The number of isolation trees to create in the forest. More trees may improve
        accuracy but also increase computation time. Default is 100.

    subsample_size : int, optional
        The size of the random subsamples to be used for constructing each isolation
        tree. A smaller subsample size can result in more robust trees but may require
        more trees in the forest. Default is 256.

    Returns:
    --------
    list
        A list of isolation trees that collectively form the Isolation Forest. Each
        tree is represented as a nested dictionary structure and can be used for
        anomaly score calculation.
    """
    trees = []

    for i in range(n_trees):
        # Create random subsamples and construct isolation trees
        subsample = data[np.random.choice(data.shape[0], subsample_size, replace=False)]
        itree = isolation_tree(subsample)
        trees.append(itree)

    return trees

# Function to calculate anomaly score for a single data point
def anomaly_score(tree, point, current_height=0):
    """
    Calculate the anomaly score for a single data point using an isolation tree.

    Anomaly scores are used to measure the degree of isolation of a data point within
    the isolation tree. Higher scores indicate anomalies, while lower scores suggest
    that the data point is normal.

    Parameters:
    -----------
    tree : dict
        A dictionary representing an isolation tree. The tree can be constructed using
        the 'isolation_tree' function. The structure includes non-leaf and leaf nodes
        with information about splitting attributes, split values, and references to
        left and right child nodes.

    point : numpy.ndarray
        The data point for which the anomaly score is calculated. It should be a
        one-dimensional numpy array with the same number of attributes as the data
        used to construct the isolation tree.

    current_height : int, optional
        The current height in the tree. This parameter is used internally for recursion
        and should not be set manually. By default, it is set to 0.

    Returns:
    --------
    float
        The anomaly score for the given data point. Higher scores indicate anomalies, while
        lower scores suggest normal data.
    """
    if "split_attribute" not in tree:
        # If it's a leaf node, return the anomaly score
        return current_height + 2 * (np.log(2 ** tree["height"] - 1) + np.euler_gamma) - (2 * (tree["height"] - 1) / (2 ** tree["height"] - 1))
    else:
        # Randomly select an attribute for splitting
        split_attribute = tree["split_attribute"]
        
        # Determine which branch to traverse based on the split attribute and value
        if point[split_attribute] < tree["split_value"]:
            return anomaly_score(tree["left"], point, current_height + 1)
        else:
            return anomaly_score(tree["right"], point, current_height + 1)
        
# Function to compute anomaly scores for the entire dataset
def isolation_forest_anomaly_score(data, trees):
    """
    Compute anomaly scores for the entire dataset using an Isolation Forest.

    Anomaly scores measure the degree of isolation of data points within the Isolation
    Forest. In this context, an anomaly is considered when the anomaly score is higher
    than some threshold, with lower scores indicating normal data points.

    Parameters:
    -----------
    data : numpy.ndarray
        The input data for which anomaly scores are calculated. It should be a
        two-dimensional numpy array where rows represent data points and columns
        represent features.

    trees : list
        A list of isolation trees that collectively form the Isolation Forest. Each tree
        is represented as a nested dictionary structure and can be used for anomaly score
        calculation.

    Returns:
    --------
    numpy.ndarray
        An array of anomaly scores for each data point in the input dataset. An anomaly is
        considered when the anomaly score is higher than some threshold, with lower scores indicating
        normal data points.
    """
    scores = np.array([anomaly_score(trees, point) for point in data])
    # normalize the anomaly scores and map them to a scale
    return 2 ** (-scores / len(trees))