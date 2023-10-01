
import pandas as pd
import numpy as np
from functions.ols_linear_reg import ols_linear_regression
from functions.ols_pred import predict_ols_linear_regression
from sklearn.metrics import mean_squared_error


def AR(df, p):
    """
    Perform Autoregressive (AR) Time Series Analysis.

    This function prepares and analyzes a time series dataset using an Autoregressive (AR) model.
    It generates lagged features, splits the dataset into training and testing sets, and applies
    linear regression to estimate coefficients for the AR model. It also calculates the Root Mean
    Squared Error (RMSE) and Mean Squared Error (MSE) to assess model performance.

    Parameters:
    - df (pandas DataFrame): The input DataFrame containing a 'Date' column and a target variable
      'AverageTemperature'.
    - p (int): The order of the Autoregressive (AR) model, representing the number of lagged
      values to consider.

    Returns:
    - List: A list containing the following elements:
        1. df_train_2 (pandas DataFrame): Training dataset with lagged features and predictions.
        2. df_test (pandas DataFrame): Testing dataset with lagged features and predictions.
        3. coef (numpy array): Coefficients (weights) of the AR model.
        4. intercept (float): Intercept (bias) of the AR model.
        5. RMSE (float): Root Mean Squared Error of the AR model predictions.
        6. MSE (float): Mean Squared Error of the AR model predictions.
        7. res (pandas DataFrame): Residuals (prediction errors) for the AR model.

    Notes:
    - The function calculates lagged values up to the order 'p' and uses them as features.
    - The dataset is split into training and testing sets, with an 80/20 split ratio.
    - Linear regression is applied to estimate coefficients for the AR model.
    - The RMSE and MSE are computed to evaluate model performance.
    - Residuals are calculated for potential use in a Moving Average (MA) model.

    Example:
    [df_train_2, df_test, coef, intercept, RMSE, MSE, res] = AR(df_temperature_data, 3)
    """
    # Copy the input DataFrame to avoid modifying the original data
    df_temp = df.copy()

    # Generate the lagged p terms
    for i in range(1, p + 1):
        df_temp['Shifted_values_%d' % i] = df_temp['AverageTemperature'].shift(i)

    # Determine the size of the training set (80% of the data)
    train_size = (int)(0.8 * df_temp.shape[0])

    # Split the data into training and testing sets
    df_train = pd.DataFrame(df_temp.iloc[0:train_size])  # Use .iloc to explicitly create a new DataFrame
    df_test = pd.DataFrame(df_temp.iloc[train_size:df_temp.shape[0]])  # Use .iloc to explicitly create a new DataFrame

    # Remove rows with missing values in the training set
    df_train_2 = df_train.dropna()

    # Extract lagged features (X) and target variable (y) for training
    X_train = df_train_2.iloc[:, 1:].values.reshape(-1, p)
    y_train = df_train_2.iloc[:, 0].values.reshape(-1, 1)

    # Apply linear regression to estimate AR model coefficients
    result = ols_linear_regression(X_train, y_train)
    coef = result[0]
    intercept = result[1]

    # Generate predictions for the training set
    df_train_2 = df_train_2.copy()
    df_train_2['Predicted_Values'] = predict_ols_linear_regression(X_train, coef, intercept)

    # Prepare lagged features for the testing set
    X_test = df_test.iloc[:, 1:].values.reshape(-1, p)

    # Generate predictions for the testing set
    df_test = df_test.copy()
    df_test['Predicted_Values'] = predict_ols_linear_regression(X_test, coef, intercept)

    # Calculate RMSE and MSE to evaluate model performance
    MSE = mean_squared_error(df_test['AverageTemperature'], df_test['Predicted_Values'])
    RMSE = np.sqrt(MSE)

    # Concatenate training and testing data for residuals calculation
    df_ar = pd.concat([df_train_2, df_test])

    # Calculate residuals
    res = pd.DataFrame()
    res['Residuals'] = df_ar['AverageTemperature'] - df_ar['Predicted_Values']

    # Print RMSE and order of the AR model
    print("The RMSE is :", RMSE, ", Value of p : ", p)

    return [df_train_2, df_test, coef, intercept, RMSE, MSE, res]