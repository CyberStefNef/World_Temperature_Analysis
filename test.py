import dash
from dash import dcc, html, Input, Output
from utils.data_loader import load_data_parquet
from functions.data_by_country import data_by_country
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

df = load_data_parquet()
df = data_by_country(df)
df = df.pivot(index="dt", columns="Country",
              values='AverageTemperature').dropna()

trend_components = {}
slopes = {}
for country in df.columns:
    result = seasonal_decompose(df[country], model='additive', period=12)
    trend_components[country] = result.trend.dropna()

all_trends = pd.concat(trend_components, axis=1)
scaler = MinMaxScaler()
all_normalized_trends = pd.DataFrame(scaler.fit_transform(
    all_trends), index=all_trends.index, columns=all_trends.columns)

# Calculate the slope based on the normalized trend
for country in all_normalized_trends.columns:
    x_data = range(len(all_normalized_trends[country]))
    y_data = all_normalized_trends[country].dropna().values

    slope, _ = np.polyfit(x_data, y_data, 1)
    slopes[country] = slope

sorted_countries = sorted(slopes, key=slopes.get, reverse=True)

# Calculate growth rate as slopes for each country
slopes_arr = np.array(list(slopes.values())).reshape(-1, 1)

# Applying KMeans clustering
n_clusters = 3  # You can change this to a different number of clusters if desired
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(slopes_arr)

# Assign cluster label to each country
clusters = {}
for idx, country in enumerate(sorted_countries):
    clusters[country] = kmeans.labels_[idx]

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Normalized Trend Component for Selected Countries'),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': f"{country} (Cluster: {clusters[country]}, Growth: {slopes[country]:.4f})", 'value': country}
                 for country in sorted_countries],
        value=['Canada'],
        multi=True
    ),
    dcc.Graph(id='trend-graph'),
    dcc.Graph(id='slope-graph')
])


@app.callback(
    [Output('trend-graph', 'figure'),
     Output('slope-graph', 'figure')],
    [Input('country-dropdown', 'value')]
)
def update_graph(selected_countries):
    traces = []

    for country in selected_countries:
        x_data = range(len(all_normalized_trends))
        y_data = all_normalized_trends[country].dropna()

        slope, intercept = np.polyfit(x_data, y_data, 1)
        reg_line = slope * np.array(x_data) + intercept

        traces.append(
            go.Scatter(x=all_normalized_trends.index,
                       y=y_data, mode='lines', name=country)
        )
        traces.append(
            go.Scatter(x=all_normalized_trends.index, y=reg_line, mode='lines',
                       name=f"{country} (Linear Fit)", line=dict(dash='dash'))
        )

    # Creating cluster-colored slope graph
    colors = px.colors.qualitative.Set1
    bar_colors = [colors[clusters[country] % len(colors)]
                  for country in sorted_countries]
    slope_fig = go.Figure(
        data=[
            go.Bar(x=list(slopes.keys()), y=list(
                slopes.values()), marker=dict(color=bar_colors))
        ],
        layout=go.Layout(title="Growth Rates for Each Country by Cluster", xaxis=dict(
            title='Country'), yaxis=dict(title='Growth Rate'), xaxis_categoryorder='total descending')
    )

    return {
        'data': traces,
        'layout': go.Layout(title="Normalized Trend Component with Linear Regression for Selected Countries", xaxis=dict(title='Year'), yaxis=dict(title='Normalized Temperature Trend'))
    }, slope_fig


if __name__ == '__main__':
    app.run_server(debug=True)
