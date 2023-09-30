
import numpy as np

def ols_linear_regression(X, y):
    """
    Perform Ordinary Least Squares (OLS) Linear Regression.

    Parameters:
    - X (numpy array): Feature matrix of shape (n_samples, n_features).
    - y (numpy array): Target vector of shape (n_samples,).

    Returns:
    - coefficients (numpy array): Coefficients (weights) of the linear model.
    - intercept (float): Intercept (bias) of the linear model.
    """
    # Add a constant (intercept) term to X
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    # Calculate coefficients and intercept using the closed-form solution (OLS)
    beta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)
    intercept = beta[0]
    coefficients = beta[1:]

    return coefficients, intercept