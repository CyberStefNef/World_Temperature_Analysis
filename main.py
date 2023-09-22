from dash import Dash, html, dcc, callback, Output, Input
from utils.data_loader import load_data
from functions.data_by_country import data_by_country
import plotly.express as px

# Load your dataset
df = load_data()

df = data_by_country(df)

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1(children='Temperature Analysis', style={'textAlign': 'center'}),
    dcc.Dropdown(df['Country'].unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph-content')
])


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[df['Country'] == value]
    return px.line(dff, x='dt', y='AverageTemperature')


if __name__ == '__main__':
    app.run_server(debug=True)
