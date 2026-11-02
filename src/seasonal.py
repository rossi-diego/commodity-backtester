"""Where today's value sits in history: percentile rank and a monthly range band.

Both use clean sessions only. Flagged sessions (roll and delivery) are left out
before anything is computed, so an expiry artefact cannot set a record.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["REFERENCE_YEAR", "Reading", "latest_reading", "monthly_band", "to_reference_year"]

# A non-leap year, so every date can be drawn on one shared January-to-December axis.
REFERENCE_YEAR = 2001
LOOKBACK_YEARS = 5


@dataclass(frozen=True)
class Reading:
    """The latest clean value and its rank against the previous five years."""

    value: float
    date: pd.Timestamp
    percentile: float | None
    latest_date: pd.Timestamp
    sessions: int


def latest_reading(values: pd.Series, flagged: pd.Series) -> Reading | None:
    """Return the last clean value and the share of clean sessions in 5 years below it.

    ``latest_date`` is the last session of any kind, so the page can say when the
    reading is older than the data. The percentile is None with under one year of
    clean history, because a rank over a few months says little.
    """
    if values.empty:
        return None
    clean = values[~flagged.reindex(values.index, fill_value=True)].dropna()
    if clean.empty:
        return None
    date = pd.Timestamp(clean.index[-1])
    window = clean.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    percentile = None
    if window.index[0] <= date - pd.DateOffset(years=1):
        percentile = float((window.iloc[:-1] < clean.iloc[-1]).mean() * 100)
    return Reading(
        value=float(clean.iloc[-1]),
        date=date,
        percentile=percentile,
        latest_date=pd.Timestamp(values.index[-1]),
        sessions=len(window),
    )


def monthly_band(values: pd.Series, flagged: pd.Series, year: int) -> pd.DataFrame:
    """Low, median and high of clean sessions by calendar month over the five years before ``year``."""
    clean = values[~flagged.reindex(values.index, fill_value=True)].dropna()
    years = pd.DatetimeIndex(clean.index).year
    past = clean[(years >= year - LOOKBACK_YEARS) & (years < year)]
    grouped = past.groupby(pd.DatetimeIndex(past.index).month)
    band = pd.DataFrame({"low": grouped.min(), "median": grouped.median(), "high": grouped.max()})
    band.index.name = "month"
    return band.reindex(range(1, 13))


def to_reference_year(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map dates onto ``REFERENCE_YEAR``. 29 February becomes 28 February."""
    days = [
        min(day, 28) if month == 2 else day
        for month, day in zip(index.month, index.day, strict=True)
    ]
    return pd.DatetimeIndex(
        [
            pd.Timestamp(REFERENCE_YEAR, month, day)
            for month, day in zip(index.month, days, strict=True)
        ]
    )
