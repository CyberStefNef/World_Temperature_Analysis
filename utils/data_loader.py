""" Data loader """

import pandas as pd


def load_data():
    """ Loading data from the data directory """

    return pd.read_csv('data/GlobalLandTemperaturesByCity.csv', parse_dates=['dt'],  dtype={'AverageTemperatureUncertainty': float, 'City': 'category', 'Country': 'category'})
