"""The README figures script: every key present, finite, and logged rather than printed."""

from __future__ import annotations

import datetime
import logging
import math
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import readme_figures
from src import config
from src.data_loader import LoadResult
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
        "risk_var_usd",
        "risk_es_usd",
        "risk_share_oil",
        "risk_share_meal",
        "risk_share_beans",
    }


def test_compute_returns_every_figure(synthetic_prices, seed) -> None:
    figures = readme_figures.compute(synthetic_prices, seed)
    assert set(figures) == _expected_keys(synthetic_prices.index.max().year)
    for name, value in figures.items():
        if isinstance(value, float | int) and not isinstance(value, bool):
            assert math.isfinite(value), name
        else:
            assert isinstance(value, str | bool), name
    assert figures["btc_usd_per_m3"] == 264
    assert 0 <= figures["top_moves_flagged"] <= 20


def test_compute_without_rins(synthetic_prices) -> None:
    figures = readme_figures.compute(synthetic_prices, None)
    assert "d4_week" not in figures
    assert "latest_after_rin" not in figures
    assert figures[f"gap_after_{readme_figures.FIRST_YEAR}"] is None


def test_main_logs_every_figure(synthetic_prices, seed, caplog, capsys) -> None:
    loaded = LoadResult(synthetic_prices, datetime.date.today(), False, None)
    with (
        patch.object(readme_figures, "load_prices", return_value=loaded),
        patch.object(readme_figures, "load_d4", return_value=RinResult(seed, "seed")),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main()
    lines = [r.getMessage() for r in caplog.records if "event=readme_figure " in r.getMessage()]
    assert len(lines) == len(_expected_keys(synthetic_prices.index.max().year))
    assert any("name=btc_usd_per_m3 value=264" in line for line in lines)
    assert capsys.readouterr().out == ""


def test_main_logs_a_load_failure(caplog) -> None:
    failed = LoadResult(None, None, False, "offline")
    with (
        patch.object(readme_figures, "load_prices", return_value=failed),
        caplog.at_level(logging.INFO, logger=readme_figures.__name__),
    ):
        readme_figures.main()
    assert any("event=readme_figures_failed" in r.getMessage() for r in caplog.records)
