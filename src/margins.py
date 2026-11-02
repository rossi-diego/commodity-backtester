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
    SHORT_TONS_PER_TONNE,
)
from .rolls import crush_flags, gap_flags

__all__ = ["CRUSH_COLUMNS", "GAP_COLUMNS", "crush_table", "gap_table", "to_metric"]

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
