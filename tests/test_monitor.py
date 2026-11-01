"""Unit tests for src/monitor.py: ratio construction and rolling statistics."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.monitor import PCT_WINDOW, Z_WINDOW, ratio_series, ratio_snapshot, rolling_stats


def _series(values: np.ndarray, start: str = "2015-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


@pytest.fixture()
def noisy_ratio() -> pd.Series:
    rng = np.random.default_rng(11)
    return _series(2.5 + rng.normal(0, 0.1, 2000).cumsum() / 20)


class TestRatioSeries:
    def test_ratio_formula(self, price_df: pd.DataFrame, tons_conv: dict[str, float]) -> None:
        """Ratio = (price_A * factor_A) / (price_B * factor_B)."""
        ratio = ratio_series(price_df, "Soybean", "Corn", tons_conv)
        expected = (price_df["Soybean"] * tons_conv["Soybean"]) / (
            price_df["Corn"] * tons_conv["Corn"]
        )
        pd.testing.assert_series_equal(ratio, expected)

    def test_ratio_positive(self, price_df: pd.DataFrame, tons_conv: dict[str, float]) -> None:
        ratio = ratio_series(price_df, "Soybean", "Corn", tons_conv)
        assert (ratio > 0).all(), "Ratio must be strictly positive for positive prices."

    @pytest.mark.parametrize("bad_leg", ["Soybean", "Corn"])
    def test_negative_and_nan_rows_are_excluded(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        bad_leg: str,
    ) -> None:
        df = price_df.copy()
        bad_negative, bad_nan = df.index[10], df.index[20]
        df.loc[bad_negative, bad_leg] = -37.63
        df.loc[bad_nan, bad_leg] = np.nan

        ratio = ratio_series(df, "Soybean", "Corn", tons_conv)

        assert bad_negative not in ratio.index
        assert bad_nan not in ratio.index
        assert ratio.notna().all()
        assert len(ratio) == len(df) - 2

    def test_z_has_no_nan_after_negative_price(self, tons_conv: dict[str, float]) -> None:
        """Regression: a NaN row kept in the ratio blinds the z-score for 252 sessions."""
        dates = pd.date_range("2015-01-01", periods=800, freq="B")
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {"Soybean": 500 + rng.normal(0, 5, 800), "Corn": 400 + rng.normal(0, 5, 800)},
            index=dates,
        )
        df.iloc[400, df.columns.get_loc("Corn")] = -37.63

        z = rolling_stats(ratio_series(df, "Soybean", "Corn", tons_conv))["z"]

        assert z.iloc[Z_WINDOW:].notna().all()


class TestRollingStats:
    def test_mean_and_z_by_hand(self) -> None:
        values = np.arange(1.0, Z_WINDOW + 1.0)
        stats = rolling_stats(_series(values))
        expected_mean = values.mean()
        expected_z = (values[-1] - expected_mean) / values.std(ddof=1)
        assert stats["mean_1y"].iloc[-1] == pytest.approx(expected_mean)
        assert stats["z"].iloc[-1] == pytest.approx(expected_z)

    def test_warmup_is_nan(self) -> None:
        stats = rolling_stats(_series(np.arange(1.0, Z_WINDOW + 2.0)))
        assert stats["z"].iloc[: Z_WINDOW - 1].isna().all()
        assert stats["z"].iloc[Z_WINDOW - 1 :].notna().all()


class TestRatioSnapshot:
    def test_values_by_hand(self, noisy_ratio: pd.Series) -> None:
        snapshot = ratio_snapshot(noisy_ratio)
        z_window = noisy_ratio.iloc[-Z_WINDOW:]
        pct_window = noisy_ratio.iloc[-PCT_WINDOW:]

        assert snapshot.level == noisy_ratio.iloc[-1]
        assert snapshot.as_of == noisy_ratio.index[-1].date()
        assert snapshot.z == pytest.approx(
            (z_window.iloc[-1] - z_window.mean()) / z_window.std(ddof=1)
        )
        assert snapshot.percentile == pytest.approx(pct_window.rank(pct=True).iloc[-1] * 100)

    def test_highest_value_is_top_percentile(self) -> None:
        snapshot = ratio_snapshot(_series(np.arange(1.0, 400.0)))
        assert snapshot.percentile == pytest.approx(100.0)
        assert snapshot.z is not None
        assert snapshot.z > 0

    def test_short_series_gives_none(self) -> None:
        snapshot = ratio_snapshot(_series(np.arange(1.0, Z_WINDOW - 1.0)))
        assert snapshot.level == Z_WINDOW - 2.0
        assert isinstance(snapshot.as_of, datetime.date)
        assert snapshot.z is None
        assert snapshot.percentile is None

    def test_constant_series_gives_none_z(self) -> None:
        snapshot = ratio_snapshot(_series(np.full(400, 2.0)))
        assert snapshot.z is None

    def test_truncation_invariance(self, noisy_ratio: pd.Series) -> None:
        """Statistics at T do not change when data after T exists: no look-ahead."""
        cut = noisy_ratio.index[1500]
        truncated = rolling_stats(noisy_ratio.loc[:cut])
        full = rolling_stats(noisy_ratio).loc[:cut]
        pd.testing.assert_frame_equal(truncated, full)

        as_of_cut = ratio_snapshot(noisy_ratio.loc[:cut])
        window = noisy_ratio.loc[:cut].iloc[-Z_WINDOW:]
        assert as_of_cut.level == noisy_ratio.loc[cut]
        assert as_of_cut.z == pytest.approx((window.iloc[-1] - window.mean()) / window.std(ddof=1))
