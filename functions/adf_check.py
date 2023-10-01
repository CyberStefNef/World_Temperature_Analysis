from statsmodels.tsa.stattools import adfuller

def adf_check(time_series, autolag):
    """
    Perform Augmented Dickey-Fuller Test (ADF) on a given time series to check for stationarity.

    Args:
        time_series (pandas.Series or array-like): The time series data to be tested for stationarity.
        autolag (str or None): This parameter determines the lag length to include in the ADF test.
            - If 'AIC' (Akaike Information Criterion), the lag length is chosen to minimize AIC.
            - If 'BIC' (Bayesian Information Criterion), the lag length is chosen to minimize BIC.
            - If None, a fixed maximum lag length is used.

    Returns:
        None: This function prints the results of the ADF test, including the test statistic, p-value,
              number of lags used, and number of observations used.

    The Augmented Dickey-Fuller Test (ADF) is used to assess whether a time series is stationary or not. 
    Stationarity is a critical assumption for many time series forecasting models. This function calculates 
    and prints the results of the ADF test, including the test statistic, p-value, and other relevant information.

    Parameters:
    - time_series: The time series data to be tested for stationarity.
    - autolag: The method for determining the number of lags in the ADF test. Use 'AIC', 'BIC', or None.

    The function then prints the ADF test results and provides a conclusion based on the p-value:
    - If the p-value is less than or equal to 0.05, there is strong evidence against the null hypothesis, 
      indicating that the data has no unit root and is stationary.
    - If the p-value is greater than 0.05, there is weak evidence against the null hypothesis, indicating 
      that the time series has a unit root, indicating it is non-stationary.

    Example:
    >>> adf_check(df.AverageTemperature, "AIC")
    """
    # Perform the Augmented Dickey-Fuller Test
    result = adfuller(time_series, autolag=autolag)

    # Print the ADF test report
    print('Augmented Dickey-Fuller Test:')
    
    # Define labels for the ADF test result values
    labels = ['ADF Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used']

    # Print each ADF test result along with its label
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    # Check the p-value and provide a conclusion
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary.")
    else:
        print("Weak evidence against the null hypothesis, time series has a unit root, indicating it is non-stationary.\n")