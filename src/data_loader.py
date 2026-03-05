"""Data acquisition layer — fetches OHLCV data from Yahoo Finance."""

from __future__ import annotations

import datetime

import pandas as pd
import yfinance as yf

from .constants import commodities_dict, tickers


class DataLoaderError(Exception):
    """Raised when data cannot be fetched or is unusable."""


def yahoo_quotes(
    start_date: str | datetime.date = "2000-01-01",
    end_date: str | datetime.date | None = None,
) -> tuple[pd.DataFrame, datetime.date | None]:
    """Download daily closing prices for all tracked commodity futures.

    Raises
    ------
    DataLoaderError
        If the network request fails or returns no usable data.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        raw: pd.DataFrame = yf.download(
            tickers,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False,
        )["Close"]
    except Exception as exc:
        raise DataLoaderError(f"Yahoo Finance request failed: {exc}") from exc

    if raw.empty:
        raise DataLoaderError(
            f"No data returned for {tickers} between {start_date} and {end_date}."
        )

    raw = raw.dropna(how="all")
    raw = raw.rename(columns=lambda col: commodities_dict.get(col, col))
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    first_available_date: datetime.date | None = (
        raw.index.min().date() if not raw.empty else None
    )

    return raw, first_available_date
