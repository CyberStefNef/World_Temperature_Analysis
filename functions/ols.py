import numpy as np

class OLS:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Perform Ordinary Least Squares (OLS) Linear Regression.

        Parameters:
        - X (numpy array): Feature matrix of shape (n_samples, n_features).
        - y (numpy array): Target vector of shape (n_samples,).
        """
        # Add a constant (intercept) term to X
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate coefficients and intercept using the closed-form solution (OLS)
        beta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        """
        Make predictions using OLS Linear Regression coefficients and intercept.

        Parameters:
        - X (numpy array): Feature matrix of shape (n_samples, n_features).

        Returns:
        - y_pred (numpy array): Predicted values for the input features.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model coefficients and intercept have not been set. Please fit the model first.")
        
        # Add a constant (intercept) term to X
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate predictions
        y_pred = X_with_intercept.dot(np.insert(self.coefficients, 0, self.intercept))

        return y_pred.reshape(-1, 1)