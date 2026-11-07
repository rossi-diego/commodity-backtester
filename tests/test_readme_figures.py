"""The README figures script: every key present, finite, logged rather than printed, and
reproduced exactly from the committed snapshot."""

from __future__ import annotations

import datetime
import logging
import math
import shutil
import string
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import readme_figures
from src import config
from src.data_loader import LoadResult, read_snapshot
from src.margins import add_rins, gap_table
from src.rins import RinResult, _read


@pytest.fixture(scope="module")
def seed() -> pd.Series:
    series = _read(config.RIN_SEED)
    assert series is not None
    return series


def _expected_keys(last_year: int) -> set[str]:
    yearly = {
        f"gap_{side}_{year}"
        for year in range(readme_figures.FIRST_YEAR, last_year + 1)
        for side in ("before", "after")
    }
    return yearly | {
        "data_date",
        "d4_week",
        "d4_usd_per_rin",
        "d4_usd_per_m3",
        "btc_usd_per_m3",
        "latest_after_rin",
        "latest_after_rin_date",
        "latest_after_rin_held",
        "crush_flagged_share",
        "top_moves_flagged",
        "crush_record_low_usd_bu",
        "crush_record_low_date",
        "crush_record_low_flagged",
        "median_abs_change_off_pair",
        "median_abs_change_board_pair",
        "risk_date",
        "risk_changes",
        "risk_vol_20_usd_t",
        "risk_vol_1y_usd_t",
        "risk_vol_ewma_usd_t",
        "risk_var_usd",
        "risk_es_usd",
        "risk_share_oil",
        "risk_share_meal",
        "risk_share_beans",
        "risk_es_tail_scenarios",
        *(f"risk_worst_{rank}_{part}" for rank in (1, 2, 3) for part in ("date", "usd")),
        "risk_backtest_start",
        "risk_backtest_days",
        "risk_backtest_exceptions",
        "risk_backtest_rate_pct",
        "risk_backtest_hs_exceptions",
        "risk_backtest_hs_rate_pct",
        "risk_kupiec_p",
        "risk_christoffersen_p",
        "risk_hs_kupiec_p",
        "risk_hs_christoffersen_p",
        "risk_es_backtest_exceptions",
        "risk_es_backtest_rate_pct",
        "risk_es_kupiec_p",
        "risk_es_christoffersen_p",
        "risk_es_back_to_back",
        "risk_es_ratio",
        "risk_z2",
        "risk_z2_p",
        "risk_clean_days_8_14_months",
        "risk_kept_changes",
        "risk_excluded_changes",
        "risk_kept_sd_usd_t",
        "risk_excluded_sd_usd_t",
        "risk_combined_changes",
        "risk_combined_var_usd",
        "risk_excluded_tested",
        "risk_excluded_exceptions",
        "risk_excluded_rate_pct",
        "risk_excluded_kupiec_p",
    }


def test_compute_returns_every_figure(synthetic_prices, seed) -> None:
    figures = readme_figures.compute(synthetic_prices, seed)
    assert set(figures) == _expected_keys(synthetic_prices.index.max().year)
    for name, value in figures.items():
        if isinstance(value, float | int) and not isinstance(value, bool):
            assert math.isfinite(value), name
        elif isinstance(value, list):
            assert all(1 <= month <= 12 for month in value), name
        else:
            assert isinstance(value, str | bool), name
    assert figures["btc_usd_per_m3"] == 264
    assert 0 <= figures["top_moves_flagged"] <= 20
    for name in ("risk_vol_ewma_usd_t", "risk_kept_sd_usd_t", "risk_excluded_sd_usd_t"):
        assert figures[name] > 0, name
    assert figures["risk_backtest_exceptions"] <= figures["risk_backtest_days"]
    assert figures["risk_clean_days_8_14_months"] == sorted(
        set(figures["risk_clean_days_8_14_months"])
    )


@pytest.mark.parametrize(("p", "shown"), [(0.0004, "<0.001"), (0.049, 0.05), (0.826, 0.83)])
def test_p_values_are_rounded_or_bounded(p, shown) -> None:
    assert readme_figures._p_value(p) == shown


def test_no_backtest_gives_no_backtest_figures() -> None:
    assert readme_figures._backtest_figures(pd.Series(dtype=float), None) == {}


def test_compute_without_rins(synthetic_prices) -> None:
    figures = readme_figures.compute(synthetic_prices, None)
    assert "d4_week" not in figures
    assert "latest_after_rin" not in figures
    assert figures[f"gap_after_{readme_figures.FIRST_YEAR}"] is None


def _figure_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "event=readme_figure " in r.getMessage()]


def test_main_logs_every_figure(synthetic_prices, seed, caplog, capsys) -> None:
    loaded = LoadResult(synthetic_prices, datetime.date.today(), False, None)
    with (
        patch.object(readme_figures, "load_prices", return_value=loaded),
        patch.object(readme_figures, "load_d4", return_value=RinResult(seed, "seed")),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main([])
    lines = _figure_lines(caplog)
    assert len(lines) == len(_expected_keys(synthetic_prices.index.max().year))
    assert any("name=btc_usd_per_m3 value=264" in line for line in lines)
    assert capsys.readouterr().out == ""


def test_main_logs_a_load_failure(caplog) -> None:
    failed = LoadResult(None, None, False, "offline")
    with (
        patch.object(readme_figures, "load_prices", return_value=failed),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main([])
    assert any("event=readme_figures_failed" in r.getMessage() for r in caplog.records)


def test_main_snapshot_needs_no_live_data(caplog, capsys) -> None:
    offline = AssertionError("the snapshot run must not load live data")
    with (
        patch.object(readme_figures, "load_prices", side_effect=offline),
        patch.object(readme_figures, "load_d4", side_effect=offline),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main(["--snapshot"])
    lines = _figure_lines(caplog)
    assert len(lines) == len(EXPECTED)
    assert "event=readme_figure name=risk_var_usd value=6114" in lines
    assert capsys.readouterr().out == ""


def test_write_snapshot_round_trips(synthetic_prices, tmp_path, monkeypatch, caplog) -> None:
    rin_cache = tmp_path / "rin_d4.csv"
    shutil.copyfile(readme_figures.d4_snapshot(), rin_cache)
    monkeypatch.setattr(config, "RIN_CACHE", rin_cache)
    monkeypatch.setattr(readme_figures, "FIXTURES", tmp_path / "fixtures")
    as_of = synthetic_prices.index.max().date()
    loaded = LoadResult(synthetic_prices, as_of, False, None)
    with (
        patch.object(readme_figures, "load_prices", return_value=loaded),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main(["--write-snapshot"])
        prices_path = readme_figures.price_snapshot(as_of.isoformat())
        first = prices_path.read_bytes()
        readme_figures.main(["--write-snapshot"])
    assert prices_path.read_bytes() == first
    back = read_snapshot(prices_path)
    pd.testing.assert_frame_equal(
        back, synthetic_prices[back.columns], check_exact=True, check_freq=False, check_names=False
    )
    d4_path = readme_figures.d4_snapshot(as_of.isoformat())
    assert d4_path.read_bytes() == rin_cache.read_bytes()
    written = [r.getMessage() for r in caplog.records if "event=snapshot_written" in r.getMessage()]
    assert f"rows={len(synthetic_prices)}" in written[0]
    assert prices_path.name in written[0]


@pytest.mark.parametrize("missing", ["prices", "rins"])
def test_write_snapshot_logs_what_is_missing(
    missing, synthetic_prices, tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(config, "RIN_CACHE", tmp_path / "absent.csv")
    monkeypatch.setattr(readme_figures, "FIXTURES", tmp_path / "fixtures")
    if missing == "prices":
        loaded = LoadResult(None, None, False, "offline")
    else:
        loaded = LoadResult(synthetic_prices, synthetic_prices.index.max().date(), False, None)
    with (
        patch.object(readme_figures, "load_prices", return_value=loaded),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main(["--write-snapshot"])
    assert any("event=snapshot_failed" in r.getMessage() for r in caplog.records)
    assert not (tmp_path / "fixtures").exists()


# ---------------------------------------------------------------------------
# The committed snapshot reproduces the README
# ---------------------------------------------------------------------------

# Logged by `python -m scripts.readme_figures --snapshot`, pasted here so a change in
# any figure shows in the diff.
EXPECTED: dict[str, Any] = {
    "data_date": "2026-11-05",
    "d4_week": "2026-09-21",
    "d4_usd_per_rin": 2.09,
    "d4_usd_per_m3": 829,
    "btc_usd_per_m3": 264,
    "gap_before_2021": 604,
    "gap_after_2021": 85,
    "gap_before_2022": 471,
    "gap_after_2022": -160,
    "gap_before_2023": 385,
    "gap_after_2023": -232,
    "gap_before_2024": 236,
    "gap_after_2024": -72,
    "gap_before_2025": 361,
    "gap_after_2025": -2,
    "gap_before_2026": 368,
    "gap_after_2026": -288,
    "latest_after_rin": -743,
    "latest_after_rin_date": "2026-11-05",
    "latest_after_rin_held": True,
    "crush_flagged_share": 0.42,
    "top_moves_flagged": 19,
    "crush_record_low_usd_bu": -1.82,
    "crush_record_low_date": "2008-09-12",
    "crush_record_low_flagged": True,
    "median_abs_change_off_pair": 0.86,
    "median_abs_change_board_pair": 0.89,
    "risk_date": "2026-10-29",
    "risk_changes": 685,
    "risk_vol_20_usd_t": 3.91,
    "risk_vol_1y_usd_t": 3.66,
    "risk_vol_ewma_usd_t": 3.59,
    "risk_var_usd": 6114,
    "risk_es_usd": 8720,
    "risk_share_oil": 69,
    "risk_share_meal": 28,
    "risk_share_beans": 2,
    "risk_es_tail_scenarios": 18,
    "risk_worst_1_date": "2023-07-26",
    "risk_worst_1_usd": -11743,
    "risk_worst_2_date": "2023-03-23",
    "risk_worst_2_usd": -10079,
    "risk_worst_3_date": "2026-06-05",
    "risk_worst_3_usd": -10035,
    "risk_backtest_start": "2021-11-17",
    "risk_backtest_days": 685,
    "risk_backtest_exceptions": 33,
    "risk_backtest_rate_pct": 4.8,
    "risk_backtest_hs_exceptions": 65,
    "risk_backtest_hs_rate_pct": 9.5,
    "risk_kupiec_p": 0.83,
    "risk_christoffersen_p": 0.29,
    "risk_hs_kupiec_p": "<0.001",
    "risk_hs_christoffersen_p": "<0.001",
    "risk_es_backtest_exceptions": 14,
    "risk_es_backtest_rate_pct": 2.0,
    "risk_es_kupiec_p": 0.43,
    "risk_es_christoffersen_p": 0.03,
    "risk_es_back_to_back": 2,
    "risk_es_ratio": 0.97,
    "risk_z2": 0.2,
    "risk_z2_p": 0.84,
    "risk_clean_days_8_14_months": [2, 4, 6],
    "risk_kept_changes": 685,
    "risk_excluded_changes": 436,
    "risk_kept_sd_usd_t": 3.53,
    "risk_excluded_sd_usd_t": 4.49,
    "risk_combined_changes": 1121,
    "risk_combined_var_usd": 5451,
    "risk_excluded_tested": 436,
    "risk_excluded_exceptions": 27,
    "risk_excluded_rate_pct": 6.2,
    "risk_excluded_kupiec_p": 0.27,
}

# Rows of data/prices.parquet on 2026-11-05, when the snapshot was written.
SNAPSHOT_ROWS = 6714

# README passages built from the figures. Numbers print as the README prints them:
# thousands separators, and a typographic minus.
README_QUOTES = [
    "The last EPA week is {d4_week}, with the D4 RIN at {d4_usd_per_rin} USD: "
    "about {d4_usd_per_m3:,} USD per m³ of biodiesel",
    *(
        f"| {year} | {{gap_before_{year}:+}} | {{gap_after_{year}:+}} |"
        for year in range(readme_figures.FIRST_YEAR, 2026)
    ),
    "| 2026 (to date) | {gap_before_2026:+} | {gap_after_2026:+} |",
    "BTC, {btc_usd_per_m3} USD/m³",
    "{crush_flagged_share:.0%} of crush sessions",
    "they hold {top_moves_flagged} of the 20 largest daily moves",
    "the record low of {crush_record_low_usd_bu} USD/bu on {crush_record_low_date}",
    "(median {median_abs_change_off_pair} against {median_abs_change_board_pair} USD/t)",
    "On the clean changes to {risk_date} ({risk_changes} in 5 years; data to {data_date}), "
    "with σ_next = {risk_vol_ewma_usd_t} USD/t",
    "| {risk_vol_20_usd_t} USD/t | {risk_vol_1y_usd_t} USD/t | {risk_var_usd:,} USD "
    "| {risk_es_usd:,} USD | oil {risk_share_oil}%, meal {risk_share_meal}%, "
    "soybeans {risk_share_beans}% |",
    "Backtest from {risk_backtest_start} to {risk_date}, at 95%",
    "| Filtered HS (shown) | {risk_backtest_days} | {risk_backtest_exceptions} "
    "| {risk_backtest_rate_pct}% | {risk_kupiec_p} | {risk_christoffersen_p} |",
    "| Plain HS, same window | {risk_backtest_days} | {risk_backtest_hs_exceptions} "
    "| {risk_backtest_hs_rate_pct}% | {risk_hs_kupiec_p} | {risk_hs_christoffersen_p} |",
    "(Christoffersen p {risk_hs_christoffersen_p})",
    "With {risk_es_tail_scenarios} tail scenarios the split moves as the window rolls",
    "| {risk_backtest_days} | {risk_es_backtest_exceptions} | {risk_es_backtest_rate_pct}% "
    "| {risk_es_kupiec_p} | {risk_es_christoffersen_p} | {risk_es_ratio:.2f} | {risk_z2:.2f} "
    "| {risk_z2_p:.2f} |",
    "At 97.5%, {risk_es_back_to_back} of the {risk_es_backtest_exceptions} exceptions fall on "
    "the clean change right after another exception, so independence is rejected at that level "
    "(p {risk_es_christoffersen_p}); the 95% exceptions do not cluster (p {risk_christoffersen_p}).",
    "A third line lists the three worst clean days of the 5 years, on 1,000 t: "
    "{risk_worst_1_usd:,} USD on {risk_worst_1_date}, {risk_worst_2_usd:,} USD on "
    "{risk_worst_2_date} and {risk_worst_3_usd:,} USD on {risk_worst_3_date}.",
    "In the 5 years to {risk_date}, {risk_excluded_changes} changes stay in one contract but "
    "touch a delivery session, against {risk_kept_changes} clean ones. They are more volatile, "
    "σ {risk_excluded_sd_usd_t} against {risk_kept_sd_usd_t} USD/t, but leaving them out does "
    "not understate the VaR on this sample: on all {risk_combined_changes:,} changes the same "
    "method gives {risk_combined_var_usd:,} USD against {risk_var_usd:,}, and the clean VaR, "
    "tested on the {risk_excluded_tested} left-out changes, has {risk_excluded_exceptions} "
    "exceptions ({risk_excluded_rate_pct}%, Kupiec p {risk_excluded_kupiec_p}).",
]


class _ReadmeFormat(string.Formatter):
    def format_field(self, value: Any, format_spec: str) -> str:
        shown = super().format_field(value, format_spec)
        if isinstance(value, int | float) and not isinstance(value, bool):
            shown = shown.replace("-", "\N{MINUS SIGN}")
        return shown


def _quotes(figures: dict[str, Any]) -> list[str]:
    return [_ReadmeFormat().format(quote, **figures) for quote in README_QUOTES]


@pytest.fixture(scope="module")
def snapshot() -> pd.DataFrame:
    return read_snapshot(readme_figures.price_snapshot())


def test_snapshot_shape(snapshot) -> None:
    path = readme_figures.price_snapshot()
    assert len(snapshot) == SNAPSHOT_ROWS
    assert snapshot.index[0] == pd.Timestamp("2000-03-15")
    assert snapshot.index[-1] == pd.Timestamp(readme_figures.SNAPSHOT_DATE)
    assert snapshot.index.is_monotonic_increasing
    assert snapshot.index.is_unique
    assert path.stat().st_size < 400_000


def test_snapshot_reproduces_readme_figures(snapshot) -> None:
    d4 = _read(readme_figures.d4_snapshot())
    assert readme_figures.compute(snapshot, d4) == EXPECTED


def test_readme_quotes_the_figures() -> None:
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [quote for quote in _quotes(EXPECTED) if quote not in readme]
    assert not missing


def test_readme_claims_hold_on_the_snapshot() -> None:
    """Sentences the README states in words, not figures."""
    assert EXPECTED["risk_combined_var_usd"] < EXPECTED["risk_var_usd"]
    assert EXPECTED["risk_es_christoffersen_p"] < 0.05 <= EXPECTED["risk_christoffersen_p"]
    assert EXPECTED["risk_z2_p"] >= 0.05


def test_a_changed_figure_is_caught() -> None:
    readme = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    changed = EXPECTED | {"risk_var_usd": EXPECTED["risk_var_usd"] + 1}
    assert any(quote not in readme for quote in _quotes(changed))
    assert _ReadmeFormat().format("{v:+}", v=-160) == "\N{MINUS SIGN}160"


def test_both_gap_columns_leave_out_held_rin_sessions(synthetic_prices, seed) -> None:
    """With the RIN held for the last months, the before column covers the after's days."""
    last = synthetic_prices.index.max()
    d4 = seed.loc[: last - pd.DateOffset(months=4)]
    figures = readme_figures.compute(synthetic_prices, d4)
    gap = add_rins(gap_table(synthetic_prices), d4)
    held = gap["rin_held"].astype(bool)
    assert held.loc[str(last.year)].any()
    year = gap.loc[str(last.year)]
    kept = year[~year["flagged"].astype(bool) & ~held.loc[str(last.year)]]
    assert figures[f"gap_before_{last.year}"] == round(kept["gap"].mean())
    assert figures[f"gap_after_{last.year}"] == round(kept["after_rin"].mean())
