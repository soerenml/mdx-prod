import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_data(
    symbols: list[str],
    start: str,
    end: str,
    interval: str
):
    """
    Retrieves historical trading data for the given symbols within the specified time range and interval.

    Args:
        symbols (list[str]): List of symbols/tickers for which to retrieve data.
        start (str): Start date of the data range in the format 'YYYY-MM-DD'.
        end (str): End date of the data range in the format 'YYYY-MM-DD'.
        interval (str): Interval at which to retrieve the data (e.g., '1d' for daily, '1h' for hourly).

    Returns:
        pandas.DataFrame: DataFrame containing the historical trading data for the specified symbols.
            Columns: 'id', 'timestamp', 'close', 'volume', 'high', 'low'.
    """
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