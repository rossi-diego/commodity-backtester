"""Unit tests for src/strategy.py — signal generation and backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import BacktestError, _compute_ratio, backtest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    price_df: pd.DataFrame,
    tons_conv: dict[str, float],
    contract_sz: float,
    strategy: str = "ratio",
    up_exit: float = 1.25,
    down_entry: float = 1.10,
    **kwargs,
) -> tuple[pd.DataFrame, bool]:
    return backtest(
        backtest_strategy=strategy,
        start_date=price_df.index[0],
        end_date=price_df.index[-1],
        df=price_df.copy(),
        up_exit=up_exit,
        down_entry=down_entry,
        commodity_chosen="Soybean",
        commodity_ratio="Corn",
        tons_conversion=tons_conv,
        contract_size=contract_sz,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------


class TestComputeRatio:
    def test_ratio_formula(self, price_df: pd.DataFrame, tons_conv: dict[str, float]) -> None:
        """Ratio = (price_A * factor_A) / (price_B * factor_B)."""
        ratio = _compute_ratio(price_df, "Soybean", "Corn", tons_conv)
        expected = (price_df["Soybean"] * tons_conv["Soybean"]) / (
            price_df["Corn"] * tons_conv["Corn"]
        )
        pd.testing.assert_series_equal(ratio, expected)

    def test_ratio_positive(self, price_df: pd.DataFrame, tons_conv: dict[str, float]) -> None:
        ratio = _compute_ratio(price_df, "Soybean", "Corn", tons_conv)
        assert (ratio > 0).all(), "Ratio must be strictly positive for positive prices."


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_returns_dataframe_and_bool(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, pos_open = _run(price_df, tons_conv, contract_sz)
        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(pos_open, bool)

    def test_required_columns_present(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        if not df_trades.empty:
            assert {"trade_price", "trade_ratio", "position", "quantity", "VaR_95"}.issubset(
                df_trades.columns
            )

    def test_no_trades_when_threshold_impossible(
        self,
        flat_price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """With flat prices the ratio is constant; impossible thresholds → zero trades."""
        ratio_value = (500.0 * tons_conv["Soybean"]) / (400.0 * tons_conv["Corn"])
        df_trades, pos_open = _run(
            flat_price_df,
            tons_conv,
            contract_sz,
            up_exit=ratio_value + 10,
            down_entry=ratio_value - 10,
        )
        assert df_trades.empty
        assert pos_open is False

    def test_signal_alternation(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Positions must strictly alternate: buy, sell, buy, sell, …"""
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        if len(df_trades) < 2:
            pytest.skip("Not enough trades to check alternation.")
        signals = df_trades["position"].tolist()
        for i in range(len(signals) - 1):
            assert signals[i] != signals[i + 1], (
                f"Consecutive identical signals at indices {i}, {i + 1}: {signals[i]}"
            )

    def test_trade_prices_match_price_series(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """trade_price must equal the actual closing price on each trade date.

        With T+1 execution the trade date IS the next bar after the signal,
        so the price recorded in df_trades must match prices[trade_date].
        """
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        for date, row in df_trades.iterrows():
            expected_price = price_df.loc[date, "Soybean"]
            assert abs(float(row["trade_price"]) - float(expected_price)) < 1e-6

    def test_t1_execution_not_on_first_bar(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """The first possible trade date must be at least the second bar.

        With T+1 execution a signal on bar 0 executes at bar 1 at the
        earliest.  Bar 0 can never be a trade date.
        """
        df_trades, _ = _run(price_df, tons_conv, contract_sz, down_entry=0.5)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        first_trade_date = df_trades.index[0]
        assert first_trade_date > price_df.index[0], (
            "First trade date equals the first bar — same-bar (T+0) execution detected."
        )

    def test_var_is_non_negative(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        if not df_trades.empty:
            assert (df_trades["VaR_95"] >= 0).all()

    def test_var_uses_only_past_returns(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """VaR computed at early trade dates should differ from late trade dates
        because fewer returns are available — proving no look-ahead is used.
        """
        df_trades, _ = _run(price_df, tons_conv, contract_sz)
        if len(df_trades) < 2:
            pytest.skip("Need at least 2 trades to compare VaR over time.")
        first_var = float(df_trades["VaR_95"].iloc[0])
        last_var = float(df_trades["VaR_95"].iloc[-1])
        # They will almost certainly differ on a sinusoidal series; if they are
        # identical, the old look-ahead (single global VaR) is still in use.
        assert first_var != last_var, (
            "VaR is identical across all trade dates — suggests look-ahead bias "
            "(single global percentile instead of per-trade rolling percentile)."
        )

    def test_unknown_strategy_raises(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        with pytest.raises(BacktestError, match="Unknown strategy"):
            _run(price_df, tons_conv, contract_sz, strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# All five strategies smoke-tested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["ratio", "mean_reversion", "momentum", "breakout", "macd"])
class TestAllStrategiesSmoke:
    def test_returns_valid_trade_log(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, pos_open = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(pos_open, bool)
        if not df_trades.empty:
            assert {"trade_price", "position", "VaR_95"}.issubset(df_trades.columns)
            assert set(df_trades["position"].unique()).issubset({"buy", "sell"})

    def test_signal_alternation(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Buy and sell signals must strictly alternate for every strategy."""
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        if len(df_trades) < 2:
            pytest.skip(f"Strategy '{strategy}' produced < 2 trades on synthetic data.")
        signals = df_trades["position"].tolist()
        for i in range(len(signals) - 1):
            assert signals[i] != signals[i + 1], (
                f"[{strategy}] consecutive identical signals at positions {i}, {i + 1}: "
                f"'{signals[i]}'"
            )

    def test_trade_prices_match_closes(
        self,
        strategy: str,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Fill price must equal the closing price on the recorded trade date."""
        df_trades, _ = _run(price_df, tons_conv, contract_sz, strategy=strategy)
        for date, row in df_trades.iterrows():
            expected = float(price_df.loc[date, "Soybean"])
            actual = float(row["trade_price"])
            assert abs(actual - expected) < 1e-6, (
                f"[{strategy}] trade_price {actual} != close {expected} on {date}"
            )


# ---------------------------------------------------------------------------
# RSI implementation check
# ---------------------------------------------------------------------------


class TestRSI:
    def test_wilder_smoothing_differs_from_sma(self) -> None:
        """Wilder's EWM-RSI must produce different values than the naive SMA-RSI.

        This test would fail if _compute_rsi reverted to rolling().mean(),
        confirming the Wilder implementation is active.
        """
        from src.strategy import _compute_rsi

        rng = np.random.default_rng(0)
        prices = pd.Series(100 + rng.normal(0, 2, 200).cumsum())

        wilder_rsi = _compute_rsi(prices, period=14)

        # Naive SMA RSI (the old, incorrect implementation)
        delta = prices.diff()
        gain_sma = delta.clip(lower=0).rolling(14).mean()
        loss_sma = (-delta.clip(upper=0)).rolling(14).mean()
        rs_sma = gain_sma / loss_sma.replace(0, float("nan"))
        sma_rsi = 100 - (100 / (1 + rs_sma))

        # The two series must differ
        diff = (wilder_rsi - sma_rsi).dropna().abs()
        assert diff.mean() > 0.1, (
            "Wilder RSI and SMA RSI are identical — _compute_rsi may not be "
            "using Wilder's exponential smoothing."
        )

    def test_rsi_bounded_0_to_100(self) -> None:
        from src.strategy import _compute_rsi

        rng = np.random.default_rng(1)
        prices = pd.Series(100 + rng.normal(0, 5, 300).cumsum()).clip(lower=1)
        rsi = _compute_rsi(prices, period=14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI out of [0, 100] range."
