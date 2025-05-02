
from constants import commodities_dict, tickers

import pandas as pd
import yfinance as yf

def yahoo_quotes(start_date, end_date):
    df_yf = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    df = df_yf['Close'].copy()

    # Rename
    df = df.rename(columns=commodities_dict)

    # Order
    ativos_ativos = [commodities_dict[ticker] for ticker in tickers]
    df = df[ativos_ativos]

    # Index
    df = df.reset_index()
    df = df.set_index('Date')
    df.index.name = 'date'
    df.columns.name = None

    return df