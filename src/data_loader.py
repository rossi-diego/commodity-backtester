from constants import commodities_dict, tickers
import pandas as pd
import yfinance as yf


def yahoo_quotes(start_date=None, end_date=None):
    if start_date is None:
        start_date = "2000-01-01"
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    df = yf.download(tickers, start=start_date, end=end_date)["Close"]
    df = df.dropna(how="all")  # remove datas sem nenhum dado

    df = df.rename(columns=lambda col: commodities_dict.get(col, col))
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    if not df.empty:
        first_available_date = df.index.min().date()
    else:
        first_available_date = None

    return df, first_available_date
