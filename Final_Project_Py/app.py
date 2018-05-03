import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import time
from finance import *

app = dash.Dash()
app.title = 'UT Dallas'
app.layout = html.Div([
    html.H1(
        'BUAN6340:Programming for Data Science - J\'s Finance Explorer',
        style={'fontSize': 22}),
    dcc.Dropdown(
        id='my-dropdown',
        options=[{
            'label': 'Apple',
            'value': 'AAPL'
        }, {
            'label': 'Google',
            'value': 'GOOGL'
        }, {
            'label': 'Microsoft',
            'value': 'MSFT'
        }],
        value='AAPL'),
    dcc.Graph(id='my-graph', config={'displaylogo': False}),
    html.Table(id='my-table')
])


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    symbol = stock(selected_dropdown_value, "6M")
    symbol.prophetModel()
    training = symbol.getTrainingData()
    test = symbol.getTestData()
    forecasts = symbol.getForecasts()
    trace1 = {
        'x': training.ds,
        'y': training.y,
        'name': 'Observations (traning data)',
        'line': dict(color=('rgb(205, 12, 24)'), width=3)
    }
    trace2 = {
        'x': test.ds,
        'y': test.y,
        'name': 'Observations (test data)',
        'line': dict(color=('rgb(0, 0, 0)'), width=3)
    }
    trace3 = {
        'x': forecasts.ds,
        'y': forecasts.yhat,
        'name': 'Forecasts',
        'line': dict(color=('rgb(72, 133, 237)'), width=3)
    }
    trace4 = {
        'x': forecasts.ds,
        'y': forecasts.yhat_lower,
        'name': 'Uncertainty',
        'fill': 'tonexty',
        'fillcolor': 'rgba(244, 194, 13, 0.1)',
        'mode': 'none'
    }
    trace5 = {
        'x': forecasts.ds,
        'y': forecasts.yhat_upper,
        'name': 'Uncertainty',
        'fill': 'tonexty',
        'fillcolor': 'rgba(244, 194, 13, 0.2)',
        'showlegend': False,
        'mode': 'none'
    }
    trace6 = {
        'x': [max(training.ds), max(training.ds)],
        'y': [
            max(max(forecasts.yhat_upper), max(training.y)),
            min(min(forecasts.yhat_lower), min(training.y))
        ],
        'text': ['Prediction Start'],
        'textposition':
        'top',
        'line': {
            'color': 'rgb(50, 171, 96)',
            'width': 2,
            'dash': 'dashdot'
        },
        'showlegend':
        False,
        'textfont':
        dict(color='rgb(50, 171, 96)'),
        'mode':
        'lines+text'
    }
    layout = {
        'title': 'NASDAQ: ' + selected_dropdown_value,
        'yaxis': dict(title='Price')
    }
    return {
        'data': [trace1, trace2, trace3, trace4, trace5, trace6],
        'layout': layout
    }


@app.callback(Output('my-table', 'children'), [Input('my-dropdown', 'value')])
def update_table(selected_dropdown_value):
    symbol = stock(selected_dropdown_value, "1Y")
    symbol.prophetModel()
    elapsedTime = time.time() - startTime
    print(elapsedTime)
    table = [
        html.Tr([
            html.Th("Shares Outstanding: "),
            html.Td("$ {:,.0f}".format(symbol.shares_outstanding)),
            html.Th("Market Cap: "),
            html.Td("$ {:,.0f}".format(
                symbol.shares_outstanding * symbol.previous_close)),
            html.Th("Mean Prediction Error on Validation Dataset: "),
            html.Td("$ {:.2f}".format(symbol.cv_avg_mean_error)),
            html.Th("Mean Prediction Error on Test Dataset: "),
            html.Td("$ {:.2f}".format(symbol.avg_mean_error)),
        ])
    ]
    return table


if __name__ == '__main__':
    app.run_server()
