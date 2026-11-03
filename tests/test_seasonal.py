"""Percentile rank and monthly band on clean sessions only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.seasonal import (
    latest_reading,
    month_vs_average,
    monthly_band,
    monthly_means,
    range_52w,
)


def _series(start: str, end: str, values: np.ndarray | None = None) -> pd.Series:
    dates = pd.bdate_range(start, end)
    data = np.arange(len(dates), dtype=float) if values is None else values
    return pd.Series(data, index=dates)


def _clean(series: pd.Series) -> pd.Series:
    return pd.Series(False, index=series.index)


class TestLatestReading:
    def test_rising_series_ranks_at_100(self) -> None:
        values = _series("2018-01-01", "2026-10-27")
        reading = latest_reading(values, _clean(values))
        assert reading is not None
        assert reading.percentile == pytest.approx(100.0)
        assert reading.date == values.index[-1]

    def test_uses_the_last_clean_session(self) -> None:
        values = _series("2018-01-01", "2026-10-27")
        flagged = _clean(values)
        flagged.iloc[-3:] = True
        reading = latest_reading(values, flagged)
        assert reading.date == values.index[-4]
        assert reading.latest_date == values.index[-1]
        assert reading.value == values.iloc[-4]

    def test_flagged_values_cannot_set_the_rank(self) -> None:
        values = _series(
            "2018-01-01",
            "2026-10-27",
            np.full(len(pd.bdate_range("2018-01-01", "2026-10-27")), 10.0),
        )
        flagged = _clean(values)
        values.iloc[100:200] = 50.0
        flagged.iloc[100:200] = True
        values.iloc[-1] = 20.0
        assert latest_reading(values, flagged).percentile == pytest.approx(100.0)

    def test_window_is_five_years(self) -> None:
        values = _series("2010-01-01", "2026-10-27")
        reading = latest_reading(values, _clean(values))
        assert 1250 < reading.sessions < 1320

    def test_short_history_has_no_rank(self) -> None:
        values = _series("2026-03-02", "2026-10-27")
        assert latest_reading(values, _clean(values)).percentile is None

    def test_empty_and_all_flagged(self) -> None:
        assert latest_reading(pd.Series(dtype=float), pd.Series(dtype=bool)) is None
        values = _series("2026-01-01", "2026-02-01")
        assert latest_reading(values, ~_clean(values)) is None

    def test_unranked_sessions_can_be_read_but_not_ranked_against(self) -> None:
        values = _series("2018-01-01", "2026-10-27")
        unranked = _clean(values)
        unranked.iloc[-30:] = True
        reading = latest_reading(values, _clean(values), unranked=unranked)
        assert reading.date == values.index[-1]
        assert reading.percentile == pytest.approx(100.0)
        assert reading.sessions < latest_reading(values, _clean(values)).sessions


class TestMonthlyMeans:
    def test_one_column_per_year_twelve_months(self) -> None:
        values = _series("2020-01-01", "2026-10-27")
        values[:] = values.index.year.to_numpy(dtype=float)
        means, counts = monthly_means(values, _clean(values))
        assert list(means.index) == list(range(1, 13))
        assert list(means.columns) == list(range(2020, 2027))
        assert means.loc[6, 2023] == 2023
        assert counts.loc[6, 2023] > 15
        assert means.loc[11:, 2026].isna().all()

    def test_flagged_sessions_are_left_out(self) -> None:
        values = _series("2020-01-01", "2025-12-31")
        values[:] = 1.0
        flagged = _clean(values)
        values.loc["2023-03-01":"2023-03-10"] = 99.0
        flagged.loc["2023-03-01":"2023-03-10"] = True
        means, _ = monthly_means(values, flagged)
        assert means.loc[3, 2023] == 1.0

    def test_thin_months_are_empty_but_the_month_to_date_needs_less(self) -> None:
        values = _series("2025-01-01", "2026-10-05")
        flagged = _clean(values)
        flagged.loc["2025-03-01":"2025-03-27"] = True
        means, counts = monthly_means(values, flagged)
        assert counts.loc[3, 2025] < 5
        assert pd.isna(means.loc[3, 2025])
        assert counts.loc[10, 2026] == 3
        assert pd.notna(means.loc[10, 2026])

    def test_empty(self) -> None:
        means, counts = monthly_means(pd.Series(dtype=float), pd.Series(dtype=bool))
        assert means.empty
        assert list(counts.index) == list(range(1, 13))


class TestMonthlyBand:
    def test_uses_the_five_previous_years_only(self) -> None:
        values = _series("2015-01-01", "2026-10-27")
        values[:] = values.index.year.to_numpy(dtype=float)
        means, _ = monthly_means(values, _clean(values))
        band = monthly_band(means, 2026)
        assert (band["low"] == 2021).all()
        assert (band["high"] == 2025).all()
        assert (band["average"] == 2023).all()

    def test_month_vs_average(self) -> None:
        values = _series("2015-01-01", "2026-10-27")
        values[:] = values.index.year.to_numpy(dtype=float)
        means, _ = monthly_means(values, _clean(values))
        assert month_vs_average(means, pd.Timestamp("2026-10-27")) == (2026.0, 2023.0)
        assert month_vs_average(means, pd.Timestamp("2027-01-04")) is None
        assert month_vs_average(means, pd.Timestamp("2026-12-01")) is None


def test_range_52w_uses_clean_sessions_of_the_last_year() -> None:
    values = _series("2024-01-01", "2026-10-27")
    flagged = _clean(values)
    values.iloc[-5] = 10_000.0
    flagged.iloc[-5] = True
    low, high = range_52w(values, flagged, values.index[-1])
    assert high == values.iloc[-1]
    assert low == values.loc["2025-10-27":].iloc[0]
