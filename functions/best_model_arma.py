
from functions.arma import ARMA

def best_model_arma(df, p, q):
    
    """
    Find the best AutoRegressive Moving Average (ARMA) model order (p, q) based on Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

    Args:
        df (pandas.DataFrame): A DataFrame containing the time series data with a column named 'AverageTemperature'.
        p (int): The maximum order of the AutoRegressive (AR) model to consider.
        q (int): The maximum order of the Moving Average (MA) model to consider.

    Returns:
        tuple: A tuple containing two tuples:
            - A tuple (best_p_mse, best_q_mse) representing the best (p, q) order based on MSE.
            - A tuple (best_p_rmse, best_q_rmse) representing the best (p, q) order based on RMSE.

    This function performs an exhaustive search for the best (p, q) order for an ARMA model based on Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) evaluation metrics. It iterates through possible (p, q) combinations within the specified range and prints the results. The function returns the best (p, q) values based on both MSE and RMSE.

    Note:
    - The input DataFrame 'df' should contain a column named 'AverageTemperature'.
    - The function will print the best (p, q) values along with corresponding MSE and RMSE.
    
    """
    # Initialize variables to store the best MSE and RMSE values
    best_mse = float('inf')
    best_rmse = float('inf')
    
    # Initialize variables to store the best (p, q) orders
    best_model_mse = None
    best_model_rmse = None

    # Iterate through possible AR order values from 1 to p
    for ar_lag in range(1, p + 1):
        # Iterate through possible MA order values from 1 to q
        for ma_lag in range(1, q + 1):
            # Print the current (p, q) combination being tried
            print(f"Trying AR lag = {ar_lag}, MA lag = {ma_lag}")
            
            # Perform ARMA analysis with the current (p, q) combination and get MSE, RMSE
            [df_ar, MSE, RMSE] = ARMA(df, ar_lag, ma_lag)

            # Check if the current MSE is better than the previous best MSE
            if MSE < best_mse:
                best_mse = MSE
                best_model_mse = (ar_lag, ma_lag)

            # Check if the current RMSE is better than the previous best RMSE
            if RMSE < best_rmse:
                best_rmse = RMSE
                best_model_rmse = (ar_lag, ma_lag)

    # Print the best (p, q) orders along with corresponding MSE and RMSE
    print('The best model according to MSE is p = {}, q = {}, with MSE = {} !'.format(best_model_mse[0], best_model_mse[1], best_mse))
    print('The best model according to RMSE is p = {}, q = {}, with RMSE = {} !'.format(best_model_rmse[0], best_model_rmse[1], best_rmse))

    return best_model_mse, best_model_rmse