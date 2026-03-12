"""Unit tests for src/strategy.py — signal generation and backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategy import BacktestError, _compute_ratio, backtest


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


class TestBacktest:
    def test_returns_dataframe_and_bool(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, pos_open = backtest(
            backtest_strategy="ratio",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.30,
            down_entry=1.05,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(pos_open, bool)

    def test_required_columns_present(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.30,
            down_entry=1.05,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
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
        # Set entry below and exit above the constant ratio — never triggers
        df_trades, pos_open = backtest(
            backtest_strategy="ratio",
            start_date=flat_price_df.index[0],
            end_date=flat_price_df.index[-1],
            df=flat_price_df.copy(),
            up_exit=ratio_value + 10,
            down_entry=ratio_value - 10,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        assert df_trades.empty
        assert pos_open is False

    def test_signal_alternation(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Positions must strictly alternate: buy, sell, buy, sell, ..."""
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.25,
            down_entry=1.10,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        if len(df_trades) < 2:
            pytest.skip("Not enough trades to check alternation.")

        signals = df_trades["position"].tolist()
        for i in range(len(signals) - 1):
            assert signals[i] != signals[i + 1], (
                f"Consecutive identical signals at indices {i}, {i+1}: {signals[i]}"
            )

    def test_trade_prices_match_price_series(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """trade_price must equal the actual closing price on each trade date."""
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.25,
            down_entry=1.10,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        for date, row in df_trades.iterrows():
            expected_price = price_df.loc[date, "Soybean"]
            assert abs(float(row["trade_price"]) - float(expected_price)) < 1e-6

    def test_var_is_non_negative(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.25,
            down_entry=1.10,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        if not df_trades.empty:
            assert (df_trades["VaR_95"] >= 0).all()

    def test_unknown_strategy_raises(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        with pytest.raises(BacktestError, match="Unknown strategy"):
            backtest(
                backtest_strategy="invalid_strategy",
                start_date=price_df.index[0],
                end_date=price_df.index[-1],
                df=price_df.copy(),
                up_exit=1.25,
                down_entry=1.10,
                commodity_chosen="Soybean",
                commodity_ratio="Corn",
                tons_conversion=tons_conv,
                contract_size=contract_sz,
            )

    def test_mean_reversion_returns_trades(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Mean reversion (Bollinger Bands) is fully implemented and must return a valid trade log."""
        df_trades, pos_open = backtest(
            backtest_strategy="mean_reversion",
            start_date=price_df.index[0],
            end_date=price_df.index[-1],
            df=price_df.copy(),
            up_exit=1.25,
            down_entry=1.10,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
        )
        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(pos_open, bool)
