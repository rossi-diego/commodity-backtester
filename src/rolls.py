"""Approximate contract calendars, used to flag sessions a continuous series gets wrong.

Yahoo's ``=F`` series keeps the expiring contract until its last trading day, so
it trades through the delivery period and then jumps to the next contract. A
trader who does not take delivery would have rolled before first notice.

From mid-September to mid-December the front months are not the board crush pair
(November beans with October, then December, products; January beans with December
products). Those sessions are not flagged: measured on 2000 to 2026, their daily
changes match clean sessions, and the level differs only by a product calendar spread.

Calendars use business days without exchange holidays, so a flag can be one
session early or late around a holiday.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import BDay, BMonthEnd

__all__ = [
    "BEAN_MONTHS",
    "DIESEL_MONTHS",
    "PRODUCT_MONTHS",
    "crush_flags",
    "front_contracts",
    "gap_flags",
    "grain_first_notice",
    "grain_last_trade",
    "monthly_last_trade",
]

BEAN_MONTHS = (1, 3, 5, 7, 8, 9, 11)
PRODUCT_MONTHS = (1, 3, 5, 7, 8, 9, 10, 12)
DIESEL_MONTHS = tuple(range(1, 13))


def grain_last_trade(year: int, month: int) -> pd.Timestamp:
    """CBOT grains: the business day before the 15th of the contract month."""
    return pd.Timestamp(year, month, 15) - BDay(1)


def grain_first_notice(year: int, month: int) -> pd.Timestamp:
    """CBOT grains: the last business day of the month before the contract month."""
    return pd.Timestamp(year, month, 1) - BDay(1)


def monthly_last_trade(year: int, month: int) -> pd.Timestamp:
    """NYMEX heating oil: the last business day of the month before the contract month."""
    return pd.Timestamp(year, month, 1) - BMonthEnd(1)


def front_contracts(
    dates: pd.DatetimeIndex, months: tuple[int, ...], last_trade: object
) -> pd.DataFrame:
    """Return the front contract on each date: its year, month and last trading day.

    The front contract is the first one whose last trading day is on or after the date,
    which is how the Yahoo continuous series behaves.
    """
    if dates.empty:
        return pd.DataFrame(
            {"year": [], "month": [], "last_trade": pd.DatetimeIndex([])}, index=dates
        )
    first, last = dates.min().year - 1, dates.max().year + 2
    contracts = sorted(
        (last_trade(year, month), year, month)  # type: ignore[operator]
        for year in range(first, last + 1)
        for month in months
    )
    expiries = pd.DatetimeIndex([expiry for expiry, _, _ in contracts])
    position = expiries.searchsorted(dates.normalize(), side="left")
    chosen = [contracts[i] for i in position]
    return pd.DataFrame(
        {
            "year": [year for _, year, _ in chosen],
            "month": [month for _, _, month in chosen],
            "last_trade": [expiry for expiry, _, _ in chosen],
        },
        index=dates,
    )


def _grain_state(dates: pd.DatetimeIndex, months: tuple[int, ...]) -> pd.DataFrame:
    front = front_contracts(dates, months, grain_last_trade)
    notice = [grain_first_notice(y, m) for y, m in zip(front["year"], front["month"], strict=True)]
    front["delivery"] = dates.normalize() >= pd.DatetimeIndex(notice)
    front["switch"] = _switched(front)
    return front


def _switched(front: pd.DataFrame) -> pd.Series:
    """True on the first session of a new contract, where the series jumps."""
    key = front["year"] * 100 + front["month"]
    return key.ne(key.shift()) & key.shift().notna()


def crush_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """True where any crush leg is in its delivery period or has just switched contract."""
    beans = _grain_state(dates, BEAN_MONTHS)
    products = _grain_state(dates, PRODUCT_MONTHS)
    flagged = beans["delivery"] | beans["switch"] | products["delivery"] | products["switch"]
    return flagged.rename("flagged")


def gap_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """True where soybean oil is in delivery or either leg has just switched contract."""
    oil = _grain_state(dates, PRODUCT_MONTHS)
    diesel = front_contracts(dates, DIESEL_MONTHS, monthly_last_trade)
    flagged = oil["delivery"] | oil["switch"] | _switched(diesel)
    return flagged.rename("flagged")
