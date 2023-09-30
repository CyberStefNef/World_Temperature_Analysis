
import numpy as np


def predict_ols_linear_regression(X, coefficients, intercept):
    """
    Make predictions using OLS Linear Regression coefficients and intercept.

    Parameters:
    - X (numpy array): Feature matrix of shape (n_samples, n_features).
    - coefficients (numpy array): Coefficients (weights) of the linear model.
    - intercept (float): Intercept (bias) of the linear model.

    Returns:
    - y_pred (numpy array): Predicted values for the input features.
    """
    # Add a constant (intercept) term to X
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    # Calculate predictions
    y_pred = X_with_intercept.dot(np.insert(coefficients, 0, intercept))

    return y_pred.reshape(-1,1)