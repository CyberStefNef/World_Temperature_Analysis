import numpy as np


class ARMA:
    def __init__(self, order):
        self.p, self.d, self.q = order
        self.ar_coeffs = np.random.randn(self.p)
        self.ma_coeffs = np.random.randn(self.q)
        self.mean = 0

    def difference(self, data, d):
        """Apply differencing to make data stationary."""
        for _ in range(d):
            data = np.diff(data)
        return data

    def inverse_difference(self, history, yhat, interval=1):
        """Inverse the differencing to get original scale."""
        return yhat + history[-interval]

    def fit(self, data, iterations=100, learning_rate=0.01):
        """
        Fit the ARIMA model to the data using Gradient Descent.
        """
        # Apply differencing
        diff_data = self.difference(data, self.d)

        error = np.zeros_like(diff_data)
        for i in range(iterations):
            for t in range(len(diff_data)):
                if t >= max(self.p, self.q):
                    ar_term = np.dot(
                        self.ar_coeffs, diff_data[t-self.p:t][::-1])
                    ma_term = np.dot(self.ma_coeffs, error[t-self.q:t][::-1])
                    prediction = self.mean + ar_term + ma_term
                    error[t] = diff_data[t] - prediction

                    # Gradient descent to update coefficients
                    self.mean += learning_rate * (diff_data[t] - prediction)
                    self.ar_coeffs += learning_rate * \
                        (diff_data[t] - prediction) * \
                        diff_data[t-self.p:t][::-1]
                    self.ma_coeffs += learning_rate * \
                        (diff_data[t] - prediction) * error[t-self.q:t][::-1]

    def forecast(self, data, steps):
        """
        Forecast future values using the ARIMA model.
        """
        forecast = np.zeros(steps)
        diff_data = self.difference(data, self.d)
        error = np.zeros_like(diff_data)

        # Prepare the last observed data and error terms
        last_data = list(data[-self.p:])
        last_error = list(error[-self.q:])

        for t in range(steps):
            if len(last_data) >= self.p and len(last_error) >= self.q:
                ar_term = np.dot(self.ar_coeffs, last_data[-self.p:][::-1])
                ma_term = np.dot(self.ma_coeffs, last_error[-self.q:][::-1])
                prediction = self.mean + ar_term + ma_term

            # Inverse differencing to get forecast on original scale
            prediction = self.inverse_difference(
                data, prediction, interval=self.d)
            forecast[t] = prediction

            # Update last data and error for next prediction
            last_data.append(prediction)
            last_error.append(0)  # Assuming error for future prediction is 0

        return forecast
