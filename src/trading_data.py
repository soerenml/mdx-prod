"""
Retrieve trading data
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_data(
    symbols: list[str],
    start: str,
    end: str,
    interval: str
):
    for ticker in symbols:
        hist = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval
        )
    hist['timestamp']= pd.to_datetime(hist.index, utc=True).strftime('%Y-%m-%d')
    hist['id'] = ticker
    hist = hist[['id', 'timestamp', 'Close', 'Volume', 'High', 'Low']]
    hist.rename(
        columns={"Close": "close", "Volume": "volume",
                 "High": "high", "Low": "low"},
        inplace=True
    )
    return hist