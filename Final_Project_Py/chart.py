import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plotHistoricalData(df):

    # df = df.reset_index().rename(columns={"index": "Date"})
    # df['Date'] = pd.to_datetime(df['Date'])
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 3))
    plt.rcParams["font.size"] = 10
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        df['Date'], df['Close'], color="#4285f4", linestyle='-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('$\Price_ {\ (USD)}$')
    return ax
