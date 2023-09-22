
def data_by_country(data):
    data = data[["Country", "AverageTemperature", "dt"]
                ].groupby(by=["Country", "dt"], as_index=False).mean()
    return data
