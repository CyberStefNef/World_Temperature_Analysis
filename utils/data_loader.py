""" Data loader """

import pandas as pd
import pycountry


def load_data_csv():
    """ Loading data from the data directory """

    return pd.read_csv('data/GlobalLandTemperaturesByCity.csv', parse_dates=['dt'],  dtype={'AverageTemperatureUncertainty': float, 'City': 'category', 'Country': 'category'})


def load_data_parquet():
    """ Loading data from the data directory """
    df = pd.read_parquet('data\GlobalLandTemperaturesByCity.parquet')
    df['dt'] = df['dt'].astype('datetime64[ns]')
    df['City'] = df['City'].astype('category')
    df['Country'] = df['Country'].astype('category')

    country_mapping = {
        "Burma": "Myanmar",
        "Côte D'Ivoire": "Côte d'Ivoire",
        "Congo (Democratic Republic of the)": "Congo, the democratic republic of the",
        "Guinea Bissau": "Guinea-Bissau",
        "Laos": "Lao People's Democratic Republic",
        "Swaziland": "Eswatini",
        'Iran': 'Iran, Islamic Republic of',
        'Russia': 'Russian Federation',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Bosnia And Herzegovina': 'Bosnia and Herzegovina',
        'Syria': 'Syrian Arab Republic',
        'Tanzania': 'Tanzania, United Republic of',
        'Vietnam': 'Viet Nam',
        'Moldova': 'Moldova, Republic of',
        'Congo (Democratic Republic Of The)': 'Congo, The Democratic Republic of the',
        'Guinea-Bissau': 'Guinea-Bissau',
        'Czech Republic': 'Czechia',
        'Taiwan': 'Taiwan, Province of China',
        'Bolivia': 'Bolivia, Plurinational State of',
        'Reunion': 'Réunion',
        'Macedonia': 'North Macedonia',
        'South Korea': 'Korea, Republic of',
        "Lao People's Democratic Republic": "Lao People's Democratic Republic"
    }
    df['Country'] = df['Country'].replace(country_mapping)

    country_to_iso = {
        country.name: country.alpha_3 for country in pycountry.countries}

    df['Country_ISO'] = df['Country'].map(country_to_iso)

    return df
