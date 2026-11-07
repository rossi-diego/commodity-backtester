"""Crush margin, oil share and feedstock gap, checked by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import BEANS, BUSHELS_PER_TONNE, DIESEL, MEAL, OIL
from src.margins import (
    CRUSH_COLUMNS,
    GAP_COLUMNS,
    add_rins,
    break_even_d4,
    crush_table,
    gap_table,
    per_gallon,
    rin_per_m3,
    to_metric,
)

# The session the README quotes, and its closes in exchange units.
PINNED_DATE = "2026-10-27"
CLOSE = {BEANS: 1229.25, MEAL: 334.6, OIL: 66.71, DIESEL: 4.7752}


def _frame(rows: dict[str, list[float]], start: str = "2026-10-26") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(next(iter(rows.values()))))
    return pd.DataFrame(rows, index=dates)


@pytest.fixture()
def one_day() -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in CLOSE.items()}, index=pd.DatetimeIndex([PINNED_DATE]))


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


class TestRins:
    def test_value_per_m3_is_1_5_rins_per_gallon(self) -> None:
        assert rin_per_m3(2.0915) == pytest.approx(1.5 * 264.172 * 2.0915, rel=1e-5)
        assert rin_per_m3(2.0915) == pytest.approx(828.8, abs=0.1)

    def test_each_session_takes_its_week_and_no_further(self) -> None:
        sessions = pd.bdate_range("2026-09-21", "2026-10-09")
        table = pd.DataFrame({"gap": 100.0, "flagged": False}, index=sessions)
        d4 = pd.Series([2.0], index=pd.DatetimeIndex(["2026-09-21"]))
        out = add_rins(table, d4)
        within = out.loc[:"2026-10-04"]
        assert within["rin_value"].notna().all()
        assert out.loc["2026-10-05":, "rin_value"].isna().all()
        assert within["after_rin"].iloc[0] == pytest.approx(100.0 - rin_per_m3(2.0))

    def test_sessions_after_the_last_week_hold_its_price_and_say_so(self) -> None:
        sessions = pd.bdate_range("2026-09-21", "2026-10-30")
        table = pd.DataFrame({"gap": 100.0, "flagged": False}, index=sessions)
        d4 = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-09-14", "2026-09-21"]))
        out = add_rins(table, d4)
        assert not out.loc[:"2026-10-04", "rin_held"].any()
        assert out.loc["2026-10-05":, "rin_held"].all()
        assert (out["rin_shown"] == rin_per_m3(2.0)).all()
        assert out["after_rin"].iloc[-1] == pytest.approx(100.0 - rin_per_m3(2.0))

    def test_a_gap_inside_the_history_is_not_filled(self) -> None:
        sessions = pd.bdate_range("2026-01-05", "2026-03-27")
        table = pd.DataFrame({"gap": 100.0, "flagged": False}, index=sessions)
        d4 = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-01-05", "2026-03-23"]))
        out = add_rins(table, d4)
        assert out.loc["2026-02-02":"2026-03-20", "after_rin"].isna().all()
        assert not out["rin_held"].any()

    @pytest.mark.parametrize("gap", [85.29, -120.0, 1e-3, 829.0])
    def test_break_even_is_the_inverse_of_the_rin_value(self, gap) -> None:
        assert rin_per_m3(break_even_d4(gap)) == pytest.approx(gap, abs=1e-9)

    def test_break_even_on_the_snapshot_gap(self) -> None:
        assert break_even_d4(0.0) == 0.0
        assert break_even_d4(85.29) == pytest.approx(85.29 / (1.5 * 264.172), rel=1e-5)
        assert f"{break_even_d4(85.29):.2f}" == "0.22"
        assert break_even_d4(-10.0) < 0

    def test_per_gallon(self) -> None:
        assert per_gallon(264.172052) == pytest.approx(1.0)

    def test_without_rins_the_columns_are_empty(self) -> None:
        table = pd.DataFrame(
            {"gap": [1.0], "flagged": [False]}, index=pd.DatetimeIndex(["2026-01-05"])
        )
        out = add_rins(table, None)
        assert out["rin_value"].isna().all()
        assert out["after_rin"].isna().all()
