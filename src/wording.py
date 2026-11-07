"""Sentences and numbers as the page prints them, kept out of ``app.py`` so tests reach them.

Every function here returns plain text from values; none reads Streamlit state.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .margins import per_gallon
from .rins import RinResult
from .seasonal import LOOKBACK_YEARS, Reading, month_vs_average
from .visualization import MONTHS

__all__ = [
    "after_rin_words",
    "day",
    "latest_close_tag",
    "month_sentence",
    "previous_years",
    "rank_clause",
    "rin_note",
    "usd",
]


def usd(value: float, unit: str, signed: bool = False, decimals: int = 0) -> str:
    """``$1,234/t``. A typographic minus, as on the chart axes, never reads as a dash."""
    shown = round(value, decimals)
    sign = "−" if shown < 0 else ("+" if signed and shown > 0 else "")
    return f"{sign}${abs(value):,.{decimals}f}/{unit}"


def day(date: pd.Timestamp) -> str:
    """``5 Nov 2026``: ``date.day``, not ``%-d``, which is not portable to Windows."""
    return f"{date.day} {date:%b %Y}"


def rank_clause(reading: Reading) -> str:
    if reading.percentile is None:
        return "with too little clean history to rank"
    return f"higher than {reading.percentile:.0f}% of clean days in the past 5 years"


def previous_years(year: int) -> str:
    """The five years before ``year``, as the monthly comparisons name them."""
    return f"{year - LOOKBACK_YEARS}–{year - 1}"


def after_rin_words(value: float) -> str:
    """The gap after the RIN in words, in USD per m³ and per gallon when the oil is cheaper."""
    if value < 0:
        return (
            f"the oil costs {usd(-value, 'm³')} ({usd(-per_gallon(value), 'gal', decimals=2)}) "
            "less than heating oil plus the RIN"
        )
    return f"the oil still costs {usd(value, 'm³')} more than heating oil plus the RIN"


def rin_note(rins: RinResult, latest: pd.Timestamp) -> str:
    """The D4 week and its age; the card's detail states the price, once."""
    if rins.prices is None or rins.as_of is None:
        return ""
    week = pd.Timestamp(rins.as_of)
    weeks = max((latest - week).days // 7, 0)
    saved = ", saved copy" if rins.source == "seed" else ""
    return f"D4 week of {week.day} {week:%b} (EPA{saved}, {weeks} weeks old). "


def latest_close_tag(reading: Reading, latest_value: float) -> str:
    """The flagged close after the last clean one, shown beside the value but not ranked."""
    if reading.date == reading.latest_date:
        return ""
    when = reading.latest_date
    return f"{when.day} {when:%b} close {usd(latest_value, 't')}, roll day"


def month_sentence(
    means: pd.DataFrame, date: pd.Timestamp, latest: pd.Timestamp, fmt: Callable[[float], str]
) -> str:
    """The by-month view's sentence: this month against the same month of the past 5 years."""
    month = MONTHS[date.month - 1]
    versus = month_vs_average(means, date)
    if versus is None:
        return f"Monthly averages of clean sessions. {month} {date.year} has too few so far to compare."
    so_far = " so far" if (date.year, date.month) == (latest.year, latest.month) else ""
    return (
        f"{month} {date.year} averages {fmt(versus[0])}{so_far}, against {fmt(versus[1])} "
        f"for {month} in {previous_years(date.year)}."
    )
