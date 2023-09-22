""" Data loader """

import pandas as pd


def load_data_csv():
    """ Loading data from the data directory """

    return pd.read_csv('data/GlobalLandTemperaturesByCity.csv', parse_dates=['dt'],  dtype={'AverageTemperatureUncertainty': float, 'City': 'category', 'Country': 'category'})


def load_data_parquet():
    """ Loading data from the data directory """
    df = pd.read_parquet('data\GlobalLandTemperaturesByCity.parquet')
    df['dt'] = df['dt'].astype('datetime64[ns]')
    df['City'] = df['City'].astype('category')
    df['Country'] = df['Country'].astype('category')

    return df


