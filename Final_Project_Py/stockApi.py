from flask_restful import Resource
from finance import getPrices, getSharesOutstanding
import pandas as pd
import numpy as np


class api(Resource):
    def get(self, symbol):
        params = {'q': symbol, 'i': 86400, 'x': 'NASDAQ', 'p': '3d'}
        df = getPrices(params)
        df = df.reset_index().rename(columns={"index": "Date"})
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        shareOutstanding = getSharesOutstanding(symbol)
        data = df.loc[df['Date'] == max(df['Date'])]
        print(data)
        getDate = data['Date'].values[0].strftime('%B-%d %Y')
        getOpen = data['Open'].values[0]
        getHigh = data['High'].values[0]
        getLow = data['Low'].values[0]
        getClose = data['Close'].values[0]
        print(getClose)
        info = {
            'symbol': symbol,
            'date': getDate,
            'open': getOpen,
            'high': getHigh,
            'low': getLow,
            'close': getClose,
            'shareOutstanding': shareOutstanding
        }
        return info
