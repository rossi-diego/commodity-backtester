"""The figures quoted in README.md, computed from the same code as the page.

Run ``python -m scripts.readme_figures --snapshot``. Each figure is logged as one
``event=readme_figure`` line, rounded as the README quotes it, so every number in
the README can be traced to a line of this output.

``--snapshot`` reads the prices and D4 series committed in ``tests/fixtures``, with
no network, so the README figures can be reproduced exactly. Without a flag the
script uses the page's live data. ``--write-snapshot`` saves the live data as a new
pair of fixtures, named by the last price date.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src import config
from src.constants import BUSHELS_PER_TONNE, GALLONS_PER_M3
from src.data_loader import load_prices, read_snapshot, write_snapshot
from src.margins import add_rins, crush_table, gap_table, rin_per_m3
from src.rins import _read, load_d4
from src.risk import (
    Backtest,
    christoffersen_independence,
    clean_changes,
    kupiec_pof,
    risk_summary,
    sample_bias,
    var_forecasts,
)
from src.rolls import off_pair
from src.seasonal import LOOKBACK_YEARS, yearly_means

__all__ = ["FIRST_YEAR", "SNAPSHOT_DATE", "compute", "d4_snapshot", "main", "price_snapshot"]

logger = logging.getLogger(__name__)

FIXTURES = config.PROJECT_ROOT / "tests" / "fixtures"
# The last price date of the committed snapshot the README figures come from.
SNAPSHOT_DATE = "2026-11-05"

# The first full year of the biodiesel-driven oil market the README describes.
FIRST_YEAR = 2021
# The blenders' tax credit: 1 USD per gallon of biodiesel.
BTC_USD_PER_GALLON = 1.0
TOP_MOVES = 20


def price_snapshot(as_of: str = SNAPSHOT_DATE) -> Path:
    """The committed daily closes up to ``as_of``."""
    return FIXTURES / f"prices_{as_of}.csv.gz"


def d4_snapshot(as_of: str = SNAPSHOT_DATE) -> Path:
    """The committed weekly D4 series used with the prices up to ``as_of``."""
    return FIXTURES / f"d4_{as_of}.csv"


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

    # Both columns leave out the sessions on a held RIN price, so they cover the same days.
    before = yearly_means(gap["gap"], gap["flagged"], gap["rin_held"])
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
    low = cast(pd.Timestamp, crush["crush"].idxmin())
    figures["crush_record_low_usd_bu"] = _rounded(
        float(crush["crush"].min()) / BUSHELS_PER_TONNE, 2
    )
    figures["crush_record_low_date"] = low.date().isoformat()
    figures["crush_record_low_flagged"] = bool(flagged.loc[low])

    changes = clean_changes(crush)["crush"].abs()
    off = off_pair(pd.DatetimeIndex(changes.index)).to_numpy()
    figures["median_abs_change_off_pair"] = _rounded(float(changes[off].median()), 2)
    figures["median_abs_change_board_pair"] = _rounded(float(changes[~off].median()), 2)

    risk = risk_summary(crush)
    if risk is not None:
        figures["risk_date"] = risk.date.date().isoformat()
        figures["risk_changes"] = risk.changes
        figures["risk_vol_20_usd_t"] = _rounded(risk.vol_20, 2)
        figures["risk_vol_1y_usd_t"] = _rounded(risk.vol_1y, 2)
        figures["risk_vol_ewma_usd_t"] = _rounded(risk.vol_ewma, 2)
        figures["risk_var_usd"] = _rounded(risk.var)
        figures["risk_es_usd"] = _rounded(risk.es)
        for leg, share in risk.contributions.items():
            figures[f"risk_share_{leg}"] = _rounded(share * 100)
        figures["risk_es_tail_scenarios"] = risk.tail_scenarios
        for rank, (day, loss) in enumerate(risk.worst, start=1):
            figures[f"risk_worst_{rank}_date"] = day.date().isoformat()
            figures[f"risk_worst_{rank}_usd"] = _rounded(loss)
        clean = clean_changes(crush)
        figures |= _backtest_figures(clean["crush"], risk.backtest)
        # Days 8 to 14, when the WASDE is released: the months with a clean change then.
        days = pd.DatetimeIndex(clean.loc[risk.date - pd.DateOffset(years=LOOKBACK_YEARS) :].index)
        mid_month = (days.day >= 8) & (days.day <= 14)
        figures["risk_clean_days_8_14_months"] = sorted({int(m) for m in days.month[mid_month]})

    bias = sample_bias(crush)
    if bias is not None:
        figures["risk_kept_changes"] = bias.kept
        figures["risk_excluded_changes"] = bias.excluded
        figures["risk_kept_sd_usd_t"] = _rounded(bias.kept_sd, 2)
        figures["risk_excluded_sd_usd_t"] = _rounded(bias.excluded_sd, 2)
        figures["risk_combined_changes"] = bias.combined_changes
        figures["risk_combined_var_usd"] = _rounded(bias.combined_var)
        figures["risk_excluded_tested"] = bias.excluded_tested
        figures["risk_excluded_exceptions"] = bias.excluded_exceptions
        if bias.excluded_tested:
            rate = bias.excluded_exceptions / bias.excluded_tested
            figures["risk_excluded_rate_pct"] = _rounded(rate * 100, 1)
            figures["risk_excluded_kupiec_p"] = _p_value(bias.excluded_kupiec_p)
    return figures


def _p_value(p: float) -> float | str | None:
    return "<0.001" if p < 0.001 else _rounded(p, 2)


def _backtest_figures(changes: pd.Series, backtest: Backtest | None) -> dict[str, Any]:
    """The filtered VaR and ES backtests, and plain historical simulation on the same days."""
    if backtest is None:
        return {}
    forecasts = var_forecasts(changes)
    hs_hits = (forecasts["change"] < -forecasts["hs"]).to_numpy()
    es_hits = (forecasts["change"] < -forecasts["var975"]).to_numpy()
    return {
        "risk_backtest_start": backtest.start.date().isoformat(),
        "risk_backtest_days": backtest.days,
        "risk_backtest_exceptions": backtest.exceptions,
        "risk_backtest_rate_pct": _rounded(backtest.rate * 100, 1),
        "risk_backtest_hs_exceptions": backtest.hs_exceptions,
        "risk_backtest_hs_rate_pct": _rounded(backtest.hs_exceptions / backtest.days * 100, 1),
        "risk_kupiec_p": _p_value(backtest.kupiec_p),
        "risk_christoffersen_p": _p_value(backtest.christoffersen_p),
        "risk_hs_kupiec_p": _p_value(kupiec_pof(backtest.hs_exceptions, backtest.days)),
        "risk_hs_christoffersen_p": _p_value(christoffersen_independence(hs_hits)),
        "risk_es_backtest_exceptions": backtest.es_exceptions,
        "risk_es_backtest_rate_pct": _rounded(backtest.es_rate * 100, 1),
        "risk_es_kupiec_p": _p_value(backtest.es_kupiec_p),
        "risk_es_christoffersen_p": _p_value(backtest.es_christoffersen_p),
        # Exceptions on the clean change right after another exception.
        "risk_es_back_to_back": int((es_hits[:-1] & es_hits[1:]).sum()),
        "risk_es_ratio": _rounded(backtest.es_ratio, 2),
        "risk_z2": _rounded(backtest.z2, 2),
        "risk_z2_p": _p_value(backtest.z2_p),
    }


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m scripts.readme_figures")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--snapshot",
        action="store_true",
        help=f"compute from the fixtures committed in tests/fixtures (data to {SNAPSHOT_DATE})",
    )
    source.add_argument(
        "--write-snapshot",
        action="store_true",
        help="save the live prices and the D4 cache as fixtures named by the last price date",
    )
    return parser.parse_args(argv)


def _log_written(path: Path, rows: int) -> None:
    root = config.PROJECT_ROOT
    shown = path.relative_to(root) if path.is_relative_to(root) else path
    logger.info("event=snapshot_written path=%s rows=%d", shown.as_posix(), rows)


def _write_snapshots() -> None:
    loaded = load_prices()
    if loaded.prices is None or loaded.as_of is None:
        logger.error("event=snapshot_failed reason=%s", loaded.error)
        return
    if _read(config.RIN_CACHE) is None:
        logger.error("event=snapshot_failed reason=no D4 cache at %s", config.RIN_CACHE.name)
        return
    as_of = loaded.as_of.isoformat()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    prices_path = price_snapshot(as_of)
    write_snapshot(loaded.prices, prices_path)
    _log_written(prices_path, len(loaded.prices))
    # A byte copy, so the snapshot parses to the same D4 values as the cache.
    d4_path = d4_snapshot(as_of)
    shutil.copyfile(config.RIN_CACHE, d4_path)
    _log_written(d4_path, len(pd.read_csv(d4_path)))


def main(argv: list[str] | None = None) -> None:
    """Log each README figure, from the live data or from the committed snapshot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse(argv)
    if args.write_snapshot:
        _write_snapshots()
        return
    if args.snapshot:
        prices = read_snapshot(price_snapshot())
        d4 = _read(d4_snapshot())
        as_of, rin_source = SNAPSHOT_DATE, "snapshot"
    else:
        loaded = load_prices()
        if loaded.prices is None:
            logger.error("event=readme_figures_failed reason=%s", loaded.error)
            return
        rins = load_d4()
        prices, d4 = loaded.prices, rins.prices
        as_of, rin_source = str(loaded.as_of), rins.source
    logger.info(
        "event=readme_figures as_of=%s rin_source=%s run=%s",
        as_of,
        rin_source,
        datetime.date.today().isoformat(),
    )
    for name, value in compute(prices, d4).items():
        logger.info("event=readme_figure name=%s value=%s", name, value)


if __name__ == "__main__":
    main()
