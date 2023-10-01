from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

def acf_pacf(time_series,lags):
    """
    Generate and display the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.

    Args:
        time_series (pandas.Series or array-like): The time series data to analyze.
        lags (int): The number of lags to include in the ACF and PACF plots.

    This function creates two subplots side by side to visualize the ACF and PACF of the given time series.
    The ACF plot shows the correlation between the series and its lagged values at various lags, while the PACF
    plot displays the partial autocorrelation, which is the correlation between the series and its lagged values 
    after removing the effect of shorter lags.

    Parameters:
        time_series (pandas.Series or array-like): The time series data to analyze.
        lags (int): The number of lags to include in the ACF and PACF plots.

    Returns:
        None: This function displays the ACF and PACF plots but does not return any values.
    """
    fig, axs = plt.subplots(1,2,figsize=(15,4))
    plot_acf(time_series, lags=lags, ax=axs[0])
    plot_pacf(time_series, lags=lags, ax=axs[1])
    plt.show()