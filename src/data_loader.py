"""Data acquisition layer — fetches OHLCV data from Yahoo Finance."""

from __future__ import annotations

import datetime

import pandas as pd
import yfinance as yf

from .constants import commodities_dict, tickers


def yahoo_quotes(
    start_date: str | datetime.date = "2000-01-01",
    end_date: str | datetime.date | None = None,
) -> tuple[pd.DataFrame, datetime.date | None]:
    """Download daily closing prices for all tracked commodity futures.

    Parameters
    ----------
    start_date:
        First date to fetch (inclusive).  Defaults to ``"2000-01-01"``.
    end_date:
        Last date to fetch (inclusive).  Defaults to today.

    Returns
    -------
    df:
        DataFrame indexed by date with one column per commodity (display name).
        Rows where *all* prices are missing are dropped.
    first_available_date:
        The earliest date that has at least one valid price, or ``None`` if the
        DataFrame is empty.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    raw: pd.DataFrame = yf.download(
        tickers,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Drop rows where every ticker is NaN (e.g. holidays)
    raw = raw.dropna(how="all")

    # Rename ticker columns to human-readable names
    raw = raw.rename(columns=lambda col: commodities_dict.get(col, col))
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    first_available_date: datetime.date | None = (
        raw.index.min().date() if not raw.empty else None
    )

    return raw, first_available_date
