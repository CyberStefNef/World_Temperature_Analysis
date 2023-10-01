
import pandas as pd
import numpy as np
from functions.ols_linear_reg import ols_linear_regression
from functions.ols_pred import predict_ols_linear_regression
from sklearn.metrics import mean_squared_error


def MA(res,q):
    """
    Perform Moving Average (MA) Time Series Analysis.

    This function applies Moving Average (MA) analysis to a time series of residuals from a previous
    model (e.g., Autoregressive, AR) to capture any remaining temporal patterns. It generates lagged
    features, splits the dataset into training and testing sets, applies linear regression to estimate
    coefficients for the MA model, and calculates the Root Mean Squared Error (RMSE) for model evaluation.

    Parameters:
    - res (pandas DataFrame): The input DataFrame containing a 'Residuals' column from a previous
      model.
    - q (int): The order of the Moving Average (MA) model, representing the number of lagged
      residuals to consider.

    Returns:
    - List: A list containing the following elements:
        1. res_train_2 (pandas DataFrame): Training dataset with lagged residuals and predictions.
        2. res_test (pandas DataFrame): Testing dataset with lagged residuals and predictions.
        3. coef (numpy array): Coefficients (weights) of the MA model.
        4. intercept (float): Intercept (bias) of the MA model.
        5. RMSE (float): Root Mean Squared Error of the MA model predictions.

    Notes:
    - The function calculates lagged residuals up to the order 'q' and uses them as features.
    - The dataset is split into training and testing sets, with an 80/20 split ratio.
    - Linear regression is applied to estimate MA model coefficients.
    - The RMSE is computed to evaluate model performance.

    Example:
    [res_train_2, res_test, coef, intercept,MSE, RMSE] = MA(df_residuals,3)
    print("The RMSE is:", RMSE)
    print("The order of the MA model is 3.")
    """
    # Generate lagged residuals as features
    for i in range(1, q + 1):
        res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

    # Determine the size of the training set (80% of the data)
    train_size = (int)(0.8 * res.shape[0])

    # Split the data into training and testing sets
    res_train = pd.DataFrame(res[0:train_size])
    res_test = pd.DataFrame(res[train_size:res.shape[0]])

    # Remove rows with missing values in the training set
    res_train_2 = res_train.dropna()

    # Extract lagged residuals (X) and residuals (y) for training
    X_train = res_train_2.iloc[:, 1:].values.reshape(-1, q)
    y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)

    # Apply linear regression to estimate MA model coefficients
    result = ols_linear_regression(X_train, y_train)
    coef = result[0]
    intercept = result[1]

    # Generate predictions for the training set
    res_train_2 = res_train_2.copy()
    res_train_2['Predicted_Values'] = predict_ols_linear_regression(X_train, coef, intercept)

    # Prepare lagged residuals for the testing set
    X_test = res_test.iloc[:, 1:].values.reshape(-1, q)

    # Generate predictions for the testing set
    res_test = res_test.copy()
    res_test['Predicted_Values'] = predict_ols_linear_regression(X_test, coef, intercept)

    # Calculate RMSE and MSE to evaluate model performance
    MSE = mean_squared_error(res_test['Residuals'], res_test['Predicted_Values'])
    RMSE = np.sqrt(MSE)

    # Print RMSE and order of the MA model
    print("The RMSE is :", RMSE,", Value of q : ",q)


    return [res_train_2, res_test, coef, intercept,MSE, RMSE]