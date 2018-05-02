import pandas as pd
import numpy as np
import requests
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from finance import getPrices, getSharesOutstanding
from plotly.offline import plot
import plotly.graph_objs as go
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class stock(object):
    def __init__(self, c, p):
        params = {}
        params['q'] = c
        params['i'] = "86400"
        params['x'] = "NASDAQ"
        params['p'] = p
        df = getPrices(params)
        df = df.reset_index().rename(columns={"index": "Date"})
        df['Date'] = pd.to_datetime(df['Date'], unit='D')
        self.df = df.copy()
        self.max_date = max(df['Date'])
        self.min_date = min(df['Date'])
        self.previous_close = df.loc[df['Date'] == max(df['Date'])][
            'Close'].values[0]
        self.forecast = pd.DataFrame()
        self.avg_mean_error = 0
        self.shares_outstanding = getSharesOutstanding(c)

    def getHistory(self):
        return self.df

    def prophetModel(self):
        df = self.df[['Date', 'Close']]
        df = df.rename(index=str, columns={"Date": "ds", "Close": "y"})
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        split_date = self.max_date - pd.DateOffset(days=30)
        train = df[df['ds'] <= split_date]
        test = df[df['ds'] >= split_date]
        model.fit(train)
        future = model.make_future_dataframe(periods=30, include_history=True)
        df_cv = cross_validation(model, horizon='30 days')
        cv_avg_mean_error = np.mean(abs(df_cv['y'] - df_cv['yhat']))
        self.cv_avg_mean_error = cv_avg_mean_error
        forecasts = model.predict(future)
        forecasts = forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        self.train = train.copy()
        self.test = test.copy()
        self.forecasts = forecasts.copy()
        # model.plot(forecast)
        evaluation = pd.merge(test, forecasts, on='ds', how='inner')
        avg_mean_error = np.mean(abs(evaluation['y'] - evaluation['yhat']))
        self.avg_mean_error = avg_mean_error

    def getTrainingData(self):
        return self.train

    def getForecasts(self):
        return self.forecasts

    def getTestData(self):
        return self.test
