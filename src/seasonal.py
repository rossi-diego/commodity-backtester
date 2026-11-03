"""Where today's value sits in history: percentile rank, monthly averages by year.

Everything uses clean sessions only. Flagged sessions (roll and delivery) are left
out before anything is computed, so an expiry artefact cannot set a record.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "LOOKBACK_YEARS",
    "Reading",
    "latest_reading",
    "month_vs_average",
    "monthly_band",
    "monthly_means",
    "range_52w",
]

LOOKBACK_YEARS = 5
# A month-year needs this many clean sessions to get an average; the month in
# progress needs fewer, and is drawn as "to date".
MIN_SESSIONS = 5
MIN_SESSIONS_TO_DATE = 3


@dataclass(frozen=True)
class Reading:
    """The latest clean value and its rank against the previous five years."""

    value: float
    date: pd.Timestamp
    percentile: float | None
    latest_date: pd.Timestamp
    sessions: int


def _clean(values: pd.Series, flagged: pd.Series) -> pd.Series:
    return values[~flagged.reindex(values.index, fill_value=True).astype(bool)].dropna()


def latest_reading(
    values: pd.Series, flagged: pd.Series, unranked: pd.Series | None = None
) -> Reading | None:
    """Return the last clean value and the share of clean sessions in 5 years below it.

    ``latest_date`` is the last session of any kind, so the page can say when the
    reading is older than the data. Sessions marked ``unranked`` can be the reading
    but are left out of the history it is ranked against. The percentile is None
    with under one year of history, because a rank over a few months says little.
    """
    if values.empty:
        return None
    clean = _clean(values, flagged)
    if clean.empty:
        return None
    date = pd.Timestamp(clean.index[-1])
    value = float(clean.iloc[-1])
    ranked = clean
    if unranked is not None:
        ranked = clean[~unranked.reindex(clean.index, fill_value=False).astype(bool)]
    window = ranked.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    percentile = None
    if not window.empty and window.index[0] <= date - pd.DateOffset(years=1):
        history = window[window.index < date]
        percentile = float((history < value).mean() * 100)
    return Reading(
        value=value,
        date=date,
        percentile=percentile,
        latest_date=pd.Timestamp(values.index[-1]),
        sessions=len(window),
    )


def monthly_means(values: pd.Series, flagged: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average of clean sessions by calendar month (rows 1 to 12) and year (columns).

    A month-year with too few clean sessions is left empty rather than averaged
    over two or three days. Also returns the session counts, for the hover.
    """
    clean = _clean(values, flagged)
    if clean.empty:
        empty = pd.DataFrame(index=pd.RangeIndex(1, 13, name="month"))
        return empty, empty
    index = pd.DatetimeIndex(clean.index)
    grouped = clean.groupby([index.month, index.year])
    means = grouped.mean().unstack()
    counts = grouped.size().unstack().fillna(0).astype(int)
    means, counts = means.reindex(range(1, 13)), counts.reindex(range(1, 13), fill_value=0)
    latest = pd.Timestamp(values.index[-1])
    minimum = pd.DataFrame(MIN_SESSIONS, index=counts.index, columns=counts.columns)
    if latest.year in minimum:
        minimum.loc[latest.month, latest.year] = MIN_SESSIONS_TO_DATE
    means = means.where(counts >= minimum)
    means.index.name = counts.index.name = "month"
    return means, counts


def monthly_band(means: pd.DataFrame, year: int) -> pd.DataFrame:
    """Low, average and high of the monthly averages over the five years before ``year``."""
    years = pd.Index(means.columns).astype(int)
    past = means.loc[:, (years >= year - LOOKBACK_YEARS) & (years < year)]
    return pd.DataFrame(
        {"low": past.min(axis=1), "average": past.mean(axis=1), "high": past.max(axis=1)},
        index=means.index,
    )


def month_vs_average(means: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float] | None:
    """The average so far in ``date``'s month, and that month's average in the five years before."""
    if date.year not in means:
        return None
    current = means[date.year].iloc[date.month - 1]
    average = monthly_band(means, date.year)["average"].iloc[date.month - 1]
    if pd.isna(current) or pd.isna(average):
        return None
    return float(current), float(average)


def range_52w(values: pd.Series, flagged: pd.Series, date: pd.Timestamp) -> tuple[float, float]:
    """Lowest and highest clean value in the year to ``date``."""
    year = _clean(values, flagged).loc[date - pd.DateOffset(years=1) : date]
    return float(year.min()), float(year.max())
