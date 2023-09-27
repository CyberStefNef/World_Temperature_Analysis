from dash import Dash, html, dcc, callback, Output, Input
from utils.data_loader import load_data_parquet
from functions.data_by_country import data_by_country
from utils.computations import compute_slopes, normalize_trend
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset and process
df = load_data_parquet()
df = data_by_country(df)
df = df.pivot(index="dt", columns="Country",
              values='AverageTemperature').dropna()
# Compute slopes
slopes = compute_slopes(df)
sorted_countries = sorted(slopes, key=slopes.get, reverse=True)

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1(children='Temperature Analysis', style={'textAlign': 'center'}),
    dcc.Dropdown(options=[{'label': country, 'value': country}
                 for country in df.columns], value='Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content'),
    dcc.Graph(id='slope-graph')
])


@callback(
    [Output('graph-content', 'figure'),
     Output('slope-graph', 'figure')],
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[value].dropna()  # dff is a Series now

    result = seasonal_decompose(dff, model='additive', period=12)
    # Assuming your function accepts a Series and returns a flattened array
    normalized_trend = normalize_trend(result.trend.dropna())

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=result.trend.index[result.trend.notna()],
                                   y=normalized_trend, mode='lines', name=value))
    trend_fig.update_layout(title="Normalized Trend Component")

    slope_fig = px.bar(x=sorted_countries, y=[slopes[country] for country in sorted_countries],
                       orientation='h', title="Rate of Temperature Change (Slope)")

    return trend_fig, slope_fig


if __name__ == '__main__':
    app.run_server(debug=True)
