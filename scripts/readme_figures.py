"""The figures quoted in README.md, computed from the same code as the page.

Run ``python -m scripts.readme_figures``. Each figure is logged as one
``event=readme_figure`` line, rounded as the README quotes it, so every number in
the README can be traced to a line of this output.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, cast

import pandas as pd

from src.constants import BUSHELS_PER_TONNE, GALLONS_PER_M3
from src.data_loader import load_prices
from src.margins import add_rins, crush_table, gap_table, rin_per_m3
from src.rins import load_d4
from src.risk import clean_changes
from src.rolls import off_pair
from src.seasonal import yearly_means

__all__ = ["FIRST_YEAR", "compute", "main"]

logger = logging.getLogger(__name__)

# The first full year of the biodiesel-driven oil market the README describes.
FIRST_YEAR = 2021
# The blenders' tax credit: 1 USD per gallon of biodiesel.
BTC_USD_PER_GALLON = 1.0
TOP_MOVES = 20


def _rounded(value: float, decimals: int = 0) -> float | int | None:
    if pd.isna(value):
        return None
    return int(round(value)) if decimals == 0 else round(float(value), decimals)


def compute(prices: pd.DataFrame, d4: pd.Series | None) -> dict[str, Any]:
    """Every README figure, keyed by name, from daily closes and the weekly D4 series."""
    crush = crush_table(prices)
    gap = add_rins(gap_table(prices), d4)
    latest = pd.Timestamp(prices.index.max())
    figures: dict[str, Any] = {"data_date": latest.date().isoformat()}

    if d4 is not None and not d4.empty:
        week = pd.Timestamp(d4.index[-1])
        figures["d4_week"] = week.date().isoformat()
        price = float(d4.iloc[-1])
        figures["d4_usd_per_rin"] = _rounded(price, 2)
        figures["d4_usd_per_m3"] = _rounded(cast(float, rin_per_m3(price)))
    figures["btc_usd_per_m3"] = _rounded(BTC_USD_PER_GALLON * GALLONS_PER_M3)

    before = yearly_means(gap["gap"], gap["flagged"])
    after = yearly_means(
        gap["after_rin"].where(gap["rin_value"].notna()), gap["flagged"], gap["rin_held"]
    )
    for year in range(FIRST_YEAR, latest.year + 1):
        figures[f"gap_before_{year}"] = _rounded(before.get(year, float("nan")))
        figures[f"gap_after_{year}"] = _rounded(after.get(year, float("nan")))

    clean_after = gap["after_rin"].where(~gap["flagged"].astype(bool)).dropna()
    if not clean_after.empty:
        date = pd.Timestamp(clean_after.index[-1])
        figures["latest_after_rin"] = _rounded(float(clean_after.iloc[-1]))
        figures["latest_after_rin_date"] = date.date().isoformat()
        figures["latest_after_rin_held"] = bool(gap.loc[date, "rin_held"])

    flagged = crush["flagged"].astype(bool)
    figures["crush_flagged_share"] = _rounded(float(flagged.mean()), 2)
    moves = crush["crush"].diff().abs().dropna()
    largest = moves.nlargest(TOP_MOVES).index
    figures["top_moves_flagged"] = int(flagged.loc[largest].sum())
    low = pd.Timestamp(crush["crush"].idxmin())  # type: ignore[arg-type]
    figures["crush_record_low_usd_bu"] = _rounded(
        float(crush["crush"].min()) / BUSHELS_PER_TONNE, 2
    )
    figures["crush_record_low_date"] = low.date().isoformat()
    figures["crush_record_low_flagged"] = bool(flagged.loc[low])

    changes = clean_changes(crush)["crush"].abs()
    off = off_pair(pd.DatetimeIndex(changes.index)).to_numpy()
    figures["median_abs_change_off_pair"] = _rounded(float(changes[off].median()), 2)
    figures["median_abs_change_board_pair"] = _rounded(float(changes[~off].median()), 2)
    return figures


def main() -> None:
    """Load the page's data and log each README figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loaded = load_prices()
    if loaded.prices is None:
        logger.error("event=readme_figures_failed reason=%s", loaded.error)
        return
    rins = load_d4()
    logger.info(
        "event=readme_figures as_of=%s rin_source=%s run=%s",
        loaded.as_of,
        rins.source,
        datetime.date.today().isoformat(),
    )
    for name, value in compute(loaded.prices, rins.prices).items():
        logger.info("event=readme_figure name=%s value=%s", name, value)


if __name__ == "__main__":
    main()
