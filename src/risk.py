"""Daily changes between consecutive clean sessions, and the 1-day risk they imply.

A change is kept only when both of its ends are clean sessions, so a roll jump or
a delivery-period move never counts as a change. The differences are taken on the
full table in row order, never after dropping flagged rows, which would join two
clean sessions across a roll.

The risk is a historical simulation on a long board crush of ``NOTIONAL_TONNES``
(long meal and oil, short soybeans): each clean change over the past
``LOOKBACK_YEARS`` is one scenario of the next day's P&L. Front-month closes, not
a hedge book, and no square-root-of-time scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .seasonal import LOOKBACK_YEARS

__all__ = [
    "ES_LEVEL",
    "MIN_CHANGES",
    "NOTIONAL_TONNES",
    "SHORT_VOL_SESSIONS",
    "VAR_LEVEL",
    "RiskSummary",
    "clean_changes",
    "risk_summary",
]

NOTIONAL_TONNES: Final = 1_000
VAR_LEVEL: Final = 0.95
ES_LEVEL: Final = 0.975
# About one year of clean changes: fewer leave too few tail scenarios for a 97.5% ES.
MIN_CHANGES: Final = 250
SHORT_VOL_SESSIONS: Final = 20

_LEGS = ["crush", "meal_value", "oil_value", "beans"]


@dataclass(frozen=True)
class RiskSummary:
    """1-day risk of a long board crush, from clean daily changes.

    ``vol_20`` and ``vol_1y`` are daily standard deviations in USD/t, not annualised.
    ``var`` and ``es`` are in USD per ``NOTIONAL_TONNES``, positive for a loss.
    ``contributions`` splits the crush variance by leg and sums to 1; a leg that
    offsets the others is negative.
    """

    date: pd.Timestamp
    vol_20: float
    vol_1y: float
    var: float
    es: float
    contributions: dict[str, float]
    changes: int


def clean_changes(table: pd.DataFrame) -> pd.DataFrame:
    """Session-to-session changes in USD/t where the session and the one before are clean.

    ``table`` is ``margins.crush_table`` output. Returns ``crush``, ``meal`` and ``oil``
    changes, and ``beans`` as minus the change in soybean cost (the short leg), so
    ``crush == meal + oil + beans`` on every row.
    """
    columns = ["crush", "meal", "oil", "beans"]
    if table.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    flagged = table["flagged"].astype(bool)
    keep = ~flagged & ~flagged.shift(1, fill_value=True)
    diff = table[_LEGS].astype(float).diff()
    changes = pd.DataFrame(
        {
            "crush": diff["crush"],
            "meal": diff["meal_value"],
            "oil": diff["oil_value"],
            "beans": -diff["beans"],
        },
        index=table.index,
    )
    return changes[keep].dropna()


def risk_summary(table: pd.DataFrame) -> RiskSummary | None:
    """Volatility, VaR, ES and risk by leg over the five years to the last clean change.

    VaR is minus the 5% quantile of the P&L (numpy's linear interpolation) and ES is
    minus the mean of the P&L at or below the 2.5% quantile. A leg's contribution is
    its covariance with the crush over the crush variance. Returns None with fewer
    than ``MIN_CHANGES`` changes in the window, or when the crush never moves.
    """
    changes = clean_changes(table)
    if changes.empty:
        return None
    date = pd.Timestamp(changes.index[-1])
    window = changes.loc[date - pd.DateOffset(years=LOOKBACK_YEARS) : date]
    crush = window["crush"]
    variance = float(crush.var())
    if len(window) < MIN_CHANGES or not variance > 0:
        return None
    pnl = NOTIONAL_TONNES * crush.to_numpy()
    tail = np.quantile(pnl, 1 - ES_LEVEL)
    year = crush.loc[date - pd.DateOffset(years=1) : date]
    return RiskSummary(
        date=date,
        vol_20=float(crush.iloc[-SHORT_VOL_SESSIONS:].std()),
        vol_1y=float(year.std()),
        var=float(-np.quantile(pnl, 1 - VAR_LEVEL)),
        es=float(-pnl[pnl <= tail].mean()),
        contributions={
            leg: float(window[leg].cov(crush)) / variance for leg in ("oil", "meal", "beans")
        },
        changes=len(window),
    )
