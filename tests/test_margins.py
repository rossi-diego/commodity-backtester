"""Crush margin, oil share and feedstock gap, checked by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import BEANS, BUSHELS_PER_TONNE, DIESEL, MEAL, OIL
from src.margins import CRUSH_COLUMNS, GAP_COLUMNS, crush_table, gap_table, to_metric

# Closes on 2026-10-27, in exchange units.
CLOSE = {BEANS: 1229.25, MEAL: 334.6, OIL: 66.71, DIESEL: 4.7752}


def _frame(rows: dict[str, list[float]], start: str = "2026-10-26") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(next(iter(rows.values()))))
    return pd.DataFrame(rows, index=dates)


@pytest.fixture()
def one_day() -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in CLOSE.items()}, index=pd.DatetimeIndex(["2026-10-27"]))


class TestCrush:
    def test_matches_the_cme_formula_in_usd_per_bushel(self, one_day) -> None:
        per_bushel = 0.022 * 334.6 + 0.11 * 66.71 - 1229.25 / 100
        crush = crush_table(one_day)["crush"].iloc[0]
        assert crush / BUSHELS_PER_TONNE == pytest.approx(per_bushel, rel=1e-6)
        assert crush == pytest.approx(88.43, abs=0.01)

    def test_value_split(self, one_day) -> None:
        row = crush_table(one_day).iloc[0]
        assert row["beans"] == pytest.approx(12.2925 * BUSHELS_PER_TONNE)
        assert row["meal_value"] + row["oil_value"] - row["beans"] == pytest.approx(row["crush"])
        assert row["oil_share"] == pytest.approx(
            row["oil_value"] / (row["oil_value"] + row["meal_value"])
        )
        assert 0.49 < row["oil_share"] < 0.51

    def test_a_missing_leg_drops_the_session(self) -> None:
        frame = _frame({BEANS: [1000.0, np.nan, 1010.0], MEAL: [300.0] * 3, OIL: [45.0] * 3})
        table = crush_table(frame)
        assert len(table) == 2
        assert table["crush"].notna().all()

    def test_a_missing_column_returns_an_empty_table(self) -> None:
        table = crush_table(_frame({BEANS: [1000.0], MEAL: [300.0]}))
        assert table.empty
        assert list(table.columns) == CRUSH_COLUMNS

    def test_flags_come_from_the_calendar(self) -> None:
        frame = _frame({BEANS: [1000.0] * 3, MEAL: [300.0] * 3, OIL: [45.0] * 3}, "2025-11-13")
        assert list(crush_table(frame)["flagged"]) == [True, True, True]


class TestGap:
    def test_feedstock_and_diesel_per_m3(self, one_day) -> None:
        row = gap_table(one_day).iloc[0]
        assert row["feedstock"] == pytest.approx(0.90 * 66.71 / 100 * 2204.62, rel=1e-5)
        assert row["diesel"] == pytest.approx(4.7752 * 264.172, rel=1e-5)
        assert row["gap"] == pytest.approx(row["feedstock"] - row["diesel"])
        assert row["gap"] == pytest.approx(62.2, abs=0.1)

    def test_a_missing_column_returns_an_empty_table(self) -> None:
        table = gap_table(_frame({OIL: [45.0]}))
        assert table.empty
        assert list(table.columns) == GAP_COLUMNS


class TestMetric:
    def test_only_present_columns_are_converted(self) -> None:
        metric = to_metric(_frame({OIL: [50.0]}))
        assert list(metric.columns) == ["oil"]
        assert metric["oil"].iloc[0] == pytest.approx(1102.31, rel=1e-5)
