import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup


def getPrices(query):
    r = requests.get(
        "https://finance.google.com/finance/getprices", params=query)
    lines = r.text.splitlines()
    data = []
    index = []
    basetime = 0
    for price in lines:
        cols = price.split(",")
        if cols[0][0] == 'a':
            basetime = int(cols[0][1:])
            index.append(datetime.fromtimestamp(basetime))
            data.append([
                float(cols[4]),
                float(cols[2]),
                float(cols[3]),
                float(cols[1]),
                int(cols[5])
            ])
        elif cols[0][0].isdigit():
            date = basetime + (int(cols[0]) * int(query['i']))
            index.append(datetime.fromtimestamp(date))
            data.append([
                float(cols[4]),
                float(cols[2]),
                float(cols[3]),
                float(cols[1]),
                int(cols[5])
            ])

    return pd.DataFrame(
        data, index=index, columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def getSharesOutstanding(company):
    url = "https://finance.yahoo.com/quote/" + company + "/key-statistics/"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    for rows in soup.find_all("td"):
        for span in rows.find_all("span"):
            if span.text == "Shares Outstanding":
                numbers = float(rows.find_next_sibling("td").text[:-1])
                units = rows.find_next_sibling("td").text[-1]
                if units == "M":
                    shares_outstanding = numbers * 10e6
                elif units == "B":
                    shares_outstanding = numbers * 10e9
                else:
                    shares_outstanding = 0
    return shares_outstanding
