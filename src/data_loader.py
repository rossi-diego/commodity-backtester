"""Data acquisition layer — fetches OHLCV data from Yahoo Finance.

Caching strategy
----------------
Downloaded data is persisted to a Parquet file (``config.YFINANCE_RAW``).
On subsequent calls the cache is served directly when it covers the requested
date range, avoiding repeated network round-trips and making the app usable
offline after the first run.  The cache is invalidated (re-fetched) when the
requested end date is newer than the latest cached date.
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd
import yfinance as yf

from . import config
from .constants import commodities_dict, tickers

__all__ = ["DataLoaderError", "yahoo_quotes"]

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Raised when data cannot be fetched or is unusable."""


def yahoo_quotes(
    start_date: str | datetime.date = config.DEFAULT_START_DATE,
    end_date: str | datetime.date | None = None,
) -> tuple[pd.DataFrame, datetime.date | None]:
    """Download daily closing prices for all tracked commodity futures.

    Results are cached as Parquet.  The cache is used when it already covers
    the full requested date range; otherwise a fresh download is performed and
    the cache is overwritten.

    Parameters
    ----------
    start_date:
        First date of the requested range.
    end_date:
        Last date of the requested range (defaults to today).

    Returns
    -------
    df:
        DataFrame of daily closing prices, columns = commodity display names.
    first_available_date:
        Earliest date present in the returned DataFrame.

    Raises
    ------
    DataLoaderError
        If the network request fails or returns no usable data.
    """
    if end_date is None:
        end_date = datetime.date.today()

    req_start = (
        pd.Timestamp(str(start_date)).date()
        if not isinstance(start_date, datetime.date)
        else start_date
    )
    req_end = (
        pd.Timestamp(str(end_date)).date() if not isinstance(end_date, datetime.date) else end_date
    )

    # ------------------------------------------------------------------
    # Try Parquet cache first
    # ------------------------------------------------------------------
    cache_path = config.YFINANCE_RAW
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            cached_start = cached.index.min().date()
            cached_end = cached.index.max().date()

            # Cache is valid when it covers the full requested window.
            # Allow up to 5-calendar-day slack on the end date to avoid
            # re-fetching on weekends/holidays when markets are closed.
            if cached_start <= req_start and cached_end >= (req_end - datetime.timedelta(days=5)):
                logger.info(
                    "Serving data from Parquet cache (%s → %s).",
                    cached_start,
                    cached_end,
                )
                result = cached.copy()
                first_available = result.index.min().date() if not result.empty else None
                return result, first_available
            else:
                logger.info(
                    "Cache does not cover requested range (%s → %s); re-fetching.",
                    req_start,
                    req_end,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read Parquet cache (%s); re-fetching.", exc)

    # ------------------------------------------------------------------
    # Download from Yahoo Finance
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Persist to cache
    # ------------------------------------------------------------------
    try:
        raw.to_parquet(cache_path)
        logger.info("Parquet cache written to %s.", cache_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write Parquet cache (%s).", exc)

    first_available_date: datetime.date | None = raw.index.min().date() if not raw.empty else None
    return raw, first_available_date
