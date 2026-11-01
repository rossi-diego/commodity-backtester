"""Ratio monitor: level, rolling z-score and rolling percentile of a price ratio.

A monitor only: no signals, no trades and no P&L. Statistics always run on the
history up to the as-of date, and the display period only changes what is charted.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "PCT_WINDOW",
    "RatioSnapshot",
    "Z_WINDOW",
    "ratio_series",
    "ratio_snapshot",
    "rolling_stats",
]

Z_WINDOW: Final[int] = 252
PCT_WINDOW: Final[int] = 1260


@dataclass(frozen=True)
class RatioSnapshot:
    """Latest reading of a ratio. Statistics are None until enough history exists."""

    level: float
    as_of: datetime.date
    z: float | None
    percentile: float | None


def ratio_series(
    df: pd.DataFrame,
    a: str,
    b: str,
    tons_conversion: dict[str, float],
) -> pd.Series:
    """Return the USD/MT ratio of ``a`` over ``b``, on dates where both prices are positive.

    Rows with a missing or non-positive price are dropped rather than left as NaN,
    otherwise one bad session would blind the rolling statistics for a whole window.
    """
    x = df[a]
    y = df[b]
    ratio = (x * tons_conversion[a]) / (y * tons_conversion[b])
    return ratio.where((x > 0) & (y > 0)).dropna()


def rolling_stats(ratio: pd.Series) -> pd.DataFrame:
    """Return the trailing 1-year mean and z-score of a ratio."""
    rolling = ratio.rolling(Z_WINDOW)
    mean = rolling.mean()
    return pd.DataFrame({"mean_1y": mean, "z": (ratio - mean) / rolling.std()})


def ratio_snapshot(ratio: pd.Series) -> RatioSnapshot:
    """Return the last level of a non-empty ratio with its z-score and percentile."""
    z = float(rolling_stats(ratio.iloc[-Z_WINDOW:])["z"].iloc[-1])
    percentile_rank = (
        ratio.iloc[-PCT_WINDOW:].rolling(PCT_WINDOW, min_periods=Z_WINDOW).rank(pct=True)
    )
    percentile = float(percentile_rank.iloc[-1]) * 100
    return RatioSnapshot(
        level=float(ratio.iloc[-1]),
        as_of=pd.Timestamp(ratio.index[-1]).date(),
        z=z if np.isfinite(z) else None,
        percentile=percentile if np.isfinite(percentile) else None,
    )
