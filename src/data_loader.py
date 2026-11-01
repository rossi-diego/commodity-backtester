"""Data acquisition layer: daily closes from Yahoo Finance behind a Parquet cache.

Caching strategy
----------------
Downloaded data is persisted to a Parquet file (``config.YFINANCE_RAW``). The
cache is served when it covers the requested range, within the slack defined in
``config``. A fresh download only replaces the cache after it passes
``_response_problems``, and the write is atomic, so a partial Yahoo response can
never overwrite good data. When a download fails or is rejected and a cache
exists, the cache is returned; without one, ``DataLoaderError`` is raised.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config
from .constants import commodities_dict, tickers

__all__ = ["DataLoaderError", "LoadResult", "load_prices", "yahoo_quotes"]

logger = logging.getLogger(__name__)

MIN_ROW_SHARE_OF_CACHE = 0.9


class DataLoaderError(Exception):
    """Raised when data cannot be fetched or is unusable."""


@dataclass(frozen=True)
class LoadResult:
    """Outcome of a price load. A failure is a value, because ``st.cache_data`` drops exceptions."""

    prices: pd.DataFrame | None
    as_of: datetime.date | None
    is_stale: bool
    error: str | None


def load_prices() -> LoadResult:
    """Load the full price history without raising on data failures."""
    try:
        prices, _ = yahoo_quotes()
    except DataLoaderError as exc:
        logger.error("event=data_load_failed reason=%s", exc)
        return LoadResult(prices=None, as_of=None, is_stale=False, error=str(exc))

    as_of = prices.index.max().date()
    age_days = (datetime.date.today() - as_of).days
    return LoadResult(
        prices=prices,
        as_of=as_of,
        is_stale=age_days > config.CACHE_STALE_AFTER_DAYS,
        error=None,
    )


def yahoo_quotes(
    start_date: str | datetime.date = config.DEFAULT_START_DATE,
    end_date: str | datetime.date | None = None,
) -> tuple[pd.DataFrame, datetime.date | None]:
    """Return daily closing prices for all tracked commodity futures.

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
        If nothing usable can be downloaded and there is no cache to fall back on.
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

    cache_path = config.YFINANCE_RAW
    cached = _read_cache(cache_path)
    if cached is not None:
        cached_start = cached.index.min().date()
        cached_end = cached.index.max().date()
        starts_in_time = cached_start <= req_start + datetime.timedelta(
            days=config.CACHE_START_SLACK_DAYS
        )
        ends_in_time = cached_end >= req_end - datetime.timedelta(
            days=config.CACHE_STALE_AFTER_DAYS
        )
        if starts_in_time and ends_in_time:
            logger.info("event=data_load source=cache rows=%d as_of=%s", len(cached), cached_end)
            return cached.copy(), cached_start

    try:
        raw: pd.DataFrame = yf.download(
            tickers,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False,
        )["Close"]
    except Exception as exc:  # noqa: BLE001
        return _fall_back(cached, f"Yahoo Finance request failed: {exc}")

    if raw.empty:
        return _fall_back(
            cached, f"No data returned for {tickers} between {start_date} and {end_date}."
        )

    raw = raw.dropna(how="all")
    raw = raw.rename(columns=lambda col: commodities_dict.get(col, col))
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"

    problems = _response_problems(raw, cached, req_start)
    if problems:
        summary = "; ".join(problems)
        logger.warning("event=cache_rejected problems=%s", summary)
        return _fall_back(cached, f"Yahoo Finance response rejected: {summary}")

    _write_cache(raw, cache_path)
    as_of = raw.index.max().date()
    logger.info("event=data_load source=download rows=%d as_of=%s", len(raw), as_of)
    return raw, raw.index.min().date()


def _read_cache(cache_path: Path) -> pd.DataFrame | None:
    """Return the cached frame, or None if it is missing, empty or unreadable."""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event=cache_unreadable reason=%s", exc)
        return None
    return None if cached.empty else cached


def _response_problems(
    raw: pd.DataFrame,
    cached: pd.DataFrame | None,
    req_start: datetime.date,
) -> list[str]:
    """List the reasons a downloaded frame must not replace the cache.

    Every tracked column needs at least one value. With a cache, each column must
    keep at least 90% of its cached observations and must not end earlier. A
    column that is missing or empty in the cache imposes no constraint, so a good
    response can repair an imperfect cache. Without a cache, each column must
    start within ``COLUMN_START_SLACK_DAYS`` of the requested start.
    """
    latest_start = pd.Timestamp(req_start) + pd.Timedelta(days=config.COLUMN_START_SLACK_DAYS)
    problems: list[str] = []

    for column in commodities_dict.values():
        new = _valid_dates(raw, column)
        if new.empty:
            problems.append(f"{column} has no values")
            continue

        if cached is None:
            if new[0] > latest_start:
                problems.append(f"{column} starts after {latest_start.date()}")
            continue

        old = _valid_dates(cached, column)
        if old.empty:
            continue

        if len(new) < MIN_ROW_SHARE_OF_CACHE * len(old):
            problems.append(f"{column} has too few rows")
        if new[-1] < old[-1]:
            problems.append(f"{column} ends before the cache")

    return problems


def _valid_dates(frame: pd.DataFrame, column: str) -> pd.DatetimeIndex:
    """Return the dates with a value in ``column``, empty if the column is absent."""
    if column not in frame.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(frame.index[frame[column].notna()])


def _fall_back(cached: pd.DataFrame | None, reason: str) -> tuple[pd.DataFrame, datetime.date]:
    """Return the existing cache after a failed refresh, or raise when there is none."""
    if cached is None:
        raise DataLoaderError(reason)
    logger.warning(
        "event=data_load source=stale_cache rows=%d as_of=%s reason=%s",
        len(cached),
        cached.index.max().date(),
        reason,
    )
    return cached.copy(), cached.index.min().date()


def _write_cache(raw: pd.DataFrame, cache_path: Path) -> None:
    """Replace the cache atomically. A failed write leaves the original file untouched."""
    temp_path = cache_path.with_suffix(".tmp")
    try:
        raw.to_parquet(temp_path)
        os.replace(temp_path, cache_path)
    except Exception as exc:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        logger.warning("event=cache_write_failed reason=%s", exc)
