import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import json
import pandas as pd
from stock import *

app = dash.Dash(__name__, static_folder='assets')
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

app.title = 'UT Dallas'
app.layout = html.Div(
    [
        html.Link(href='/assets/normalize.css', rel='stylesheet'),
        html.H1(
            'Programming for Data Science - J\'s Finance Explorer',
            style={'fontSize': 22}),
        dcc.Dropdown(
            id='dropdown',
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
        html.Div(id='temp-value', style={'display': 'none'}),
        dcc.Graph(
            id='graph',
            config={
                'editable':
                False,
                'modeBarButtonsToRemove': [
                    'sendDataToCloud', 'zoom2d', 'select2d', 'pan2d',
                    'lasso2d', 'resetScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian', 'toggleSpikelines'
                ],
                'displaylogo':
                False
            }),
        html.Table(id='table', style={'fontSize': 15})
    ],
    style={
        'marginLeft': 15,
        'marginRight': 15
    })


@app.callback(Output('temp-value', 'children'), [Input('dropdown', 'value')])
def clean_data(selected_dropdown_value):
    symbol = stock(selected_dropdown_value, "1Y")
    symbol.prophetModel()
    training = symbol.getTrainingData()
    test = symbol.getTestData()
    forecasts = symbol.getForecasts()

    shareOutstanding = symbol.shares_outstanding
    marketCap = symbol.shares_outstanding * symbol.previous_close
    cvMeanError = symbol.cv_mean_error
    meanError = symbol.mean_error
    summary = pd.DataFrame(
        [[
            selected_dropdown_value, shareOutstanding, marketCap, cvMeanError,
            meanError
        ]],
        index=['0'],
        columns=[
            'company', 'shareOutstanding', 'marketCap', 'cvMeanError',
            'meanError'
        ])

    datasets = {
        'training': training.to_json(orient='split', date_format='iso'),
        'test': test.to_json(orient='split', date_format='iso'),
        'forecasts': forecasts.to_json(orient='split', date_format='iso'),
        'summary': summary.to_json(orient='split', date_format='iso')
    }
    return json.dumps(datasets)


@app.callback(Output('graph', 'figure'), [Input('temp-value', 'children')])
def update_graph(jsonified_cleaned_data):
    datasets = json.loads(jsonified_cleaned_data)
    training = pd.read_json(datasets['training'], orient='split')
    test = pd.read_json(datasets['test'], orient='split')
    forecasts = pd.read_json(datasets['forecasts'], orient='split')
    summary = pd.read_json(datasets['summary'], orient='split')
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
        'title': 'NASDAQ: ' + summary.company[0],
        'yaxis': dict(title='Price'),
        'legend': dict(x=0, y=1)
    }
    return {
        'data': [trace1, trace2, trace3, trace4, trace5, trace6],
        'layout': layout
    }


@app.callback(Output('table', 'children'), [Input('temp-value', 'children')])
def update_table(jsonified_cleaned_data):
    datasets = json.loads(jsonified_cleaned_data)
    summary = pd.read_json(datasets['summary'], orient='split')
    table = [
        html.Tr([
            html.Th("Shares Outstanding: "),
            html.Td("$ {:,.0f}".format(summary.shareOutstanding[0])),
            html.Th("Market Cap: "),
            html.Td("$ {:,.0f}".format(summary.marketCap[0])),
            html.Th("Mean Prediction Error on Validation Dataset: "),
            html.Td("$ {:.2f}".format(summary.cvMeanError[0])),
            html.Th("Mean Prediction Error on Test Dataset: "),
            html.Td("$ {:.2f}".format(summary.meanError[0]))
        ])
    ]
    return table


if __name__ == '__main__':
    app.run_server(debug=True)
