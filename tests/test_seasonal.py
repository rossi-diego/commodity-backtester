"""Percentile rank and monthly band on clean sessions only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.seasonal import REFERENCE_YEAR, latest_reading, monthly_band, to_reference_year


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


class TestMonthlyBand:
    def test_uses_the_five_previous_years_only(self) -> None:
        values = _series("2015-01-01", "2026-10-27")
        values[:] = values.index.year.to_numpy(dtype=float)
        band = monthly_band(values, _clean(values), 2026)
        assert (band["low"] == 2021).all()
        assert (band["high"] == 2025).all()
        assert list(band.index) == list(range(1, 13))

    def test_flagged_sessions_are_left_out(self) -> None:
        values = _series("2020-01-01", "2025-12-31")
        values[:] = 1.0
        flagged = _clean(values)
        values.loc["2023-03-01":"2023-03-10"] = 99.0
        flagged.loc["2023-03-01":"2023-03-10"] = True
        assert monthly_band(values, flagged, 2026).loc[3, "high"] == 1.0


class TestReferenceYear:
    def test_maps_dates_and_leap_day(self) -> None:
        dates = pd.DatetimeIndex(["2024-02-29", "2026-10-27"])
        mapped = to_reference_year(dates)
        assert list(mapped) == [
            pd.Timestamp(REFERENCE_YEAR, 2, 28),
            pd.Timestamp(REFERENCE_YEAR, 10, 27),
        ]
