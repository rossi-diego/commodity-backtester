"""Crush margin, oil share and the biodiesel feedstock gap, in metric units."""

from __future__ import annotations

import pandas as pd

from .constants import (
    BEANS,
    BUSHELS_PER_TONNE,
    DIESEL,
    GALLONS_PER_M3,
    LB_PER_TONNE,
    MEAL,
    MEAL_YIELD,
    OIL,
    OIL_PER_M3_BIODIESEL,
    OIL_YIELD,
    RINS_PER_GALLON_BIODIESEL,
    SHORT_TONS_PER_TONNE,
)
from .rolls import crush_flags, gap_flags

__all__ = [
    "CRUSH_COLUMNS",
    "GAP_COLUMNS",
    "RIN_TOLERANCE",
    "add_rins",
    "crush_table",
    "gap_table",
    "rin_per_m3",
    "to_metric",
]

# A weekly RIN price is used for the sessions of its week and the next. Past that the
# value is left empty rather than stretched.
RIN_TOLERANCE = pd.Timedelta(days=13)

CRUSH_COLUMNS = ["beans", "meal_value", "oil_value", "crush", "oil_share", "flagged"]
GAP_COLUMNS = ["feedstock", "diesel", "gap", "flagged"]


def to_metric(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert exchange quotes to USD per tonne (beans, meal, oil) and USD per m3 (diesel)."""
    metric = pd.DataFrame(index=prices.index)
    if BEANS in prices:
        metric["beans"] = prices[BEANS] / 100 * BUSHELS_PER_TONNE
    if MEAL in prices:
        metric["meal"] = prices[MEAL] * SHORT_TONS_PER_TONNE
    if OIL in prices:
        metric["oil"] = prices[OIL] / 100 * LB_PER_TONNE
    if DIESEL in prices:
        metric["diesel"] = prices[DIESEL] * GALLONS_PER_M3
    return metric


def crush_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Board crush per tonne of soybeans, on sessions where all three legs have a price.

    ``crush`` equals the CME board crush in USD per bushel times the bushels in a tonne.
    ``flagged`` marks sessions to leave out of statistics, see ``rolls.crush_flags``.
    """
    legs = [BEANS, MEAL, OIL]
    if any(leg not in prices for leg in legs):
        return pd.DataFrame(columns=CRUSH_COLUMNS)
    # A price is never carried forward into a margin: a missing leg drops the session.
    metric = to_metric(prices[legs].dropna())
    table = pd.DataFrame(index=metric.index)
    table["beans"] = metric["beans"]
    table["meal_value"] = MEAL_YIELD * metric["meal"]
    table["oil_value"] = OIL_YIELD * metric["oil"]
    table["crush"] = table["meal_value"] + table["oil_value"] - table["beans"]
    table["oil_share"] = table["oil_value"] / (table["oil_value"] + table["meal_value"])
    table["flagged"] = crush_flags(pd.DatetimeIndex(table.index)).to_numpy()
    return table


def gap_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Soybean oil feedstock cost per m3 of biodiesel against heating oil per m3."""
    legs = [OIL, DIESEL]
    if any(leg not in prices for leg in legs):
        return pd.DataFrame(columns=GAP_COLUMNS)
    metric = to_metric(prices[legs].dropna())
    table = pd.DataFrame(index=metric.index)
    table["feedstock"] = OIL_PER_M3_BIODIESEL * metric["oil"]
    table["diesel"] = metric["diesel"]
    table["gap"] = table["feedstock"] - table["diesel"]
    table["flagged"] = gap_flags(pd.DatetimeIndex(table.index)).to_numpy()
    return table


def rin_per_m3(usd_per_rin: float | pd.Series) -> float | pd.Series:
    """Value of the D4 RINs a m3 of biodiesel generates: 1.5 RINs per gallon."""
    return RINS_PER_GALLON_BIODIESEL * GALLONS_PER_M3 * usd_per_rin


def add_rins(table: pd.DataFrame, d4: pd.Series | None) -> pd.DataFrame:
    """Add the D4 RIN value per m3 and the gap after it, matched to each session's week."""
    out = table.copy()
    if d4 is None or d4.empty or table.empty:
        out["rin_value"] = float("nan")
    else:
        weekly = rin_per_m3(d4.sort_index())
        assert isinstance(weekly, pd.Series)
        sessions = pd.DatetimeIndex(table.index).as_unit("ns")
        weekly.index = pd.DatetimeIndex(weekly.index).as_unit("ns")
        out["rin_value"] = weekly.reindex(
            sessions, method="ffill", tolerance=RIN_TOLERANCE
        ).to_numpy()
    out["after_rin"] = out["gap"] - out["rin_value"]
    return out
