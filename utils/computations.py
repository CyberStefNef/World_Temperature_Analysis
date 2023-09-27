import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler


def compute_slopes(df):
    slopes = {}
    for country in df.columns:
        result = seasonal_decompose(df[country], model='additive', period=12)
        trend = result.trend.dropna()

        # Time as independent variable
        X = np.array(range(len(trend))).reshape(-1, 1)
        y = trend.values  # Trend as dependent variable

        # Calculate slope using linear regression
        reg = LinearRegression().fit(X, y)
        slopes[country] = reg.coef_[0]
    return slopes


def normalize_trend(result):
    scaler = MinMaxScaler()
    normalized_trend = scaler.fit_transform(result.values.reshape(-1, 1))
    return normalized_trend
