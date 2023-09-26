
def data_by_country(data):
    data = data[["Country", "AverageTemperature", "dt", "Country_ISO"]
                ].groupby(by=["Country", "Country_ISO", "dt"], as_index=False).mean()
    return data
