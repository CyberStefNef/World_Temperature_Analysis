

import pandas as pd
import numpy as np
from functions.ols_linear_reg import ols_linear_regression
from functions.ols_pred import predict_ols_linear_regression
from sklearn.metrics import mean_squared_error


def ARMA(df,p,q):
    
    """
    Perform an AutoRegressive Moving Average (ARMA) analysis on a time series DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame containing a time series with a column named 'AverageTemperature'.
        p (int): The order of the AutoRegressive (AR) model.
        q (int): The order of the Moving Average (MA) model.

    Returns:
        list: A list containing the following items:
            1. pandas.DataFrame: A DataFrame with additional columns, including 'Predicted_Values', 
               representing the ARMA model predictions.
            2. float: Mean Squared Error (MSE) of the ARMA model on the testing data.
            3. float: Root Mean Squared Error (RMSE) of the ARMA model on the testing data.

    This function performs an ARMA analysis on a given time series data, consisting of AR (AutoRegressive) and 
    MA (Moving Average) modeling. It first extracts the AR component by fitting a linear regression model 
    to the lagged values of the 'AverageTemperature' series. Then, it calculates the residuals and uses them 
    for fitting the MA component with another linear regression model. Finally, it combines the AR and MA 
    components to generate predictions for the entire time series and returns the resulting DataFrame, 
    along with MSE and RMSE as evaluation metrics.

    Note:
    - The function assumes that the input DataFrame 'df' contains a column named 'AverageTemperature'.
    - The input 'p' and 'q' are positive integers specifying the orders of the AR and MA models, respectively.
    - The training-testing split is performed using an 80-20 ratio.
    - Linear regression models are used for parameter estimation.
    """
    
    #AR part
    # Copy the input DataFrame to avoid modifying the original data
    df_temp = df.copy()

    # Generate the lagged p terms
    for i in range(1, p + 1):
        df_temp['Shifted_values_%d' % i] = df_temp['AverageTemperature'].shift(i)

    # Determine the size of the training set (80% of the data)
    train_size = (int)(0.8 * df_temp.shape[0])

    # Split the data into training and testing sets
    df_train = pd.DataFrame(df_temp[0:train_size])
    df_test = pd.DataFrame(df_temp[train_size:df.shape[0]])

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


    # Concatenate training and testing data for residuals calculation
    df_ar = pd.concat([df_train_2, df_test])

    # Calculate residuals
    res = pd.DataFrame()
    res['Residuals'] = df_ar['AverageTemperature'] - df_ar['Predicted_Values']
    
    
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
    result2 = ols_linear_regression(X_train, y_train)
    coef2 = result2[0]
    intercept2 = result2[1]

    # Generate predictions for the training set
    res_train_2 = res_train_2.copy()
    res_train_2['Predicted_Values'] = predict_ols_linear_regression(X_train, coef2, intercept2)

    # Prepare lagged residuals for the testing set
    X_test = res_test.iloc[:, 1:].values.reshape(-1, q)

    # Generate predictions for the testing set
    res_test = res_test.copy()
    res_test['Predicted_Values'] = predict_ols_linear_regression(X_test, coef2, intercept2)
    
    # Calculate RMSE and MSE to evaluate model performance
    MSE = mean_squared_error(res_test['Residuals'], res_test['Predicted_Values'])
    RMSE = np.sqrt(MSE)

    # Print RMSE and order of the MA model
    print("The MSE is :", MSE,", Value of p : ",p, "Value of q :",q)
    print("The RMSE is :", RMSE,", Value of p : ",p, "Value of q :",q)
    
    # Pediction
    res_c = pd.concat([res_train_2,res_test])
    
    # Adding the predicted data from res to the AR part (ARMA)
    df_ar.Predicted_Values += res_c.Predicted_Values
    
    return [df_ar,MSE,RMSE]