"""Unit tests for src/utils.py — PnL engine and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import _build_trades
from src.utils import (
    backtest_performance,
    backtest_performance_extended,
    pnl_trades,
    summarize_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_full(
    price_df: pd.DataFrame,
    tons_conv: dict[str, float],
    contract_sz: float,
    up_exit: float = 1.25,
    down_entry: float = 1.10,
) -> tuple[pd.DataFrame, dict | None, bool]:
    ratio = (price_df["Soybean"] * tons_conv["Soybean"]) / (price_df["Corn"] * tons_conv["Corn"])
    signals = (ratio < down_entry).astype(int) - (ratio > up_exit).astype(int)
    df_trades_raw, pos_open = _build_trades(
        signals=signals,
        prices=price_df["Soybean"],
        commodity_chosen="Soybean",
        tons_conversion=tons_conv,
        contract_size=contract_sz,
        stop_loss_pct=None,
        take_profit_pct=None,
    )
    df_trades, mtm = pnl_trades(
        df_trades=df_trades_raw,
        df_prices=price_df,
        commodity_chosen="Soybean",
        tons_conversion=tons_conv,
        contract_size=contract_sz,
        position_open=pos_open,
    )
    return df_trades, mtm, pos_open


def _initial_capital(price_df: pd.DataFrame, tons_conv: dict, contract_sz: float) -> float:
    first_price = float(price_df["Soybean"].dropna().iloc[0])
    return contract_sz * tons_conv["Soybean"] * first_price


# ---------------------------------------------------------------------------
# PnL engine tests
# ---------------------------------------------------------------------------


class TestPnlTrades:
    def test_empty_input_returns_empty(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        empty = pd.DataFrame(
            columns=["trade_price", "trade_ratio", "position", "quantity", "VaR_95"]
        )
        empty.index.name = "date"
        result, mtm = pnl_trades(
            df_trades=empty,
            df_prices=price_df,
            commodity_chosen="Soybean",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
            position_open=False,
        )
        assert result.empty
        assert mtm is None

    def test_pnl_columns_added(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _, _ = _run_full(price_df, tons_conv, contract_sz)
        if not df_trades.empty:
            assert "pnl_usd" in df_trades.columns
            assert "pnl_usd_cumsum" in df_trades.columns

    def test_cumsum_is_running_total(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, _, _ = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        expected_cumsum = df_trades["pnl_usd"].cumsum()
        pd.testing.assert_series_equal(
            df_trades["pnl_usd_cumsum"], expected_cumsum, check_names=False
        )

    def test_buy_rows_have_zero_pnl(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """PnL is only realised on sell rows; buy rows should be zero."""
        df_trades, _, _ = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        buy_pnl = df_trades.loc[df_trades["position"] == "buy", "pnl_usd"]
        assert (buy_pnl == 0.0).all()

    def test_pnl_calculation_manual(self, tons_conv: dict[str, float], contract_sz: float) -> None:
        """Verify PnL formula against CME contract specification.

        Contract: 5 000 bu soybeans.
        Buy at 480 c/bu, sell at 520 c/bu → price diff = 40 c/bu.

        Correct PnL = 40 c/bu × 5 000 bu ÷ 100 c/$ = $2 000.

        Equivalently in $/MT:
          pnl = (520 - 480) × tons_conv["Soybean"] × contract_sz
              = 40 × 0.36744 × 136.08 ≈ $2 000
        """
        dates = pd.date_range("2022-01-03", periods=4, freq="B")
        price_df = pd.DataFrame(
            {"Soybean": [480.0, 490.0, 510.0, 520.0], "Corn": 400.0}, index=dates
        )
        price_df.index.name = "date"

        df_trades_raw = pd.DataFrame(
            {
                "trade_price": [480.0, 520.0],
                "trade_ratio": [1.10, 1.25],
                "position": ["buy", "sell"],
                "quantity": [1, -1],
                "VaR_95": [0.0, 0.0],
            },
            index=[dates[0], dates[3]],
        )
        df_trades_raw.index.name = "date"

        df_trades, _ = pnl_trades(
            df_trades=df_trades_raw,
            df_prices=price_df,
            commodity_chosen="Soybean",
            tons_conversion=tons_conv,
            contract_size=contract_sz,
            position_open=False,
        )

        # CME spec: 5 000 bu × 40 c/bu ÷ 100 = $2 000
        expected_pnl = (520.0 - 480.0) * tons_conv["Soybean"] * contract_sz
        actual_pnl = float(df_trades.loc[dates[3], "pnl_usd"])
        assert abs(actual_pnl - expected_pnl) < 0.01, (
            f"Expected ${expected_pnl:.2f} (CME spec), got ${actual_pnl:.2f}. "
            "Check that contract_size is used directly, not multiplied by tons_conversion."
        )

    def test_mtm_none_when_no_open_position(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if not pos_open:
            assert mtm is None

    def test_mtm_present_when_open_position(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Force an open position by setting exit threshold very high."""
        df_trades, mtm, pos_open = _run_full(
            price_df, tons_conv, contract_sz, up_exit=999.0, down_entry=1.05
        )
        if pos_open and not df_trades.empty:
            assert mtm is not None
            assert "pnl_usd" in mtm
            assert "date" in mtm

    def test_slippage_reduces_pnl(self, tons_conv: dict[str, float], contract_sz: float) -> None:
        """With slippage the sell price is lower and buy price higher → lower PnL."""
        dates = pd.date_range("2022-01-03", periods=2, freq="B")
        price_df = pd.DataFrame({"Soybean": [500.0, 550.0], "Corn": 400.0}, index=dates)
        price_df.index.name = "date"
        raw = pd.DataFrame(
            {
                "trade_price": [500.0, 550.0],
                "trade_ratio": [1.1, 1.2],
                "position": ["buy", "sell"],
                "quantity": [1, -1],
                "VaR_95": [0.0, 0.0],
            },
            index=dates,
        )
        raw.index.name = "date"

        clean, _ = pnl_trades(
            raw, price_df, "Soybean", tons_conv, contract_sz, False, slippage_pct=0.0
        )
        slipped, _ = pnl_trades(
            raw.copy(), price_df, "Soybean", tons_conv, contract_sz, False, slippage_pct=0.1
        )

        assert float(slipped.iloc[-1]["pnl_usd"]) < float(clean.iloc[-1]["pnl_usd"])


# ---------------------------------------------------------------------------
# Performance metrics tests
# ---------------------------------------------------------------------------


class TestBacktestPerformance:
    def test_returns_dataframe(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        result = backtest_performance(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Metric", "Value"]

    def test_all_expected_metrics_present(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        result = backtest_performance_extended(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
        )
        metrics = set(result["Metric"].tolist())
        required = {
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Profit Factor",
            "Recovery Factor",
            "Win Rate (%)",
            "Max Drawdown (USD)",
            "Annualised Return (%)",
        }
        missing = required - metrics
        assert not missing, f"Missing metrics: {missing}"

    def test_win_rate_between_0_and_100(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        result = backtest_performance(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
        )
        wr = float(result.loc[result["Metric"] == "Win Rate (%)", "Value"].iloc[0])
        assert 0.0 <= wr <= 100.0

    def test_max_drawdown_non_negative(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        result = backtest_performance(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
        )
        mdd = float(result.loc[result["Metric"] == "Max Drawdown (USD)", "Value"].iloc[0])
        assert mdd >= 0.0

    def test_sharpe_is_finite(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Sharpe must be a finite float, not NaN or inf.

        Previous bug: equity curve started at 0, making pct_change() return
        inf on the first trade, corrupting the standard deviation and producing
        NaN Sharpe.  Fixed by normalising daily PnL by initial_capital instead
        of using pct_change on the cumulative equity curve.
        """
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)
        result = backtest_performance(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
            initial_capital=ic,
        )
        sharpe = float(result.loc[result["Metric"] == "Sharpe Ratio", "Value"].iloc[0])
        assert np.isfinite(sharpe), f"Sharpe should be finite, got {sharpe}"

    def test_calmar_is_finite_and_nonzero_when_profitable(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Calmar must not be stuck at exactly 0.

        Previous bug: _annualised_return guarded against equity.iloc[0] == 0
        (which was always true because the equity curve is cumulative PnL
        starting at 0), making it always return 0.0 and Calmar always 0.
        Fixed by using initial_capital as the denominator instead.
        """
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)
        result = backtest_performance_extended(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
            initial_capital=ic,
        )
        calmar = float(result.loc[result["Metric"] == "Calmar Ratio", "Value"].iloc[0])
        # Calmar should be finite (may be inf when drawdown == 0, which is valid)
        total_pnl = float(result.loc[result["Metric"] == "Total Profit (USD)", "Value"].iloc[0])
        if total_pnl > 0:
            # If strategy is profitable, Calmar must be a positive finite number
            # or inf (zero drawdown).
            assert calmar > 0 or not np.isfinite(calmar), (
                f"Calmar should be positive when profitable, got {calmar}"
            )

    def test_sortino_is_not_nan(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Sortino must never be NaN.

        It may be inf when there are no losing days (mathematically correct),
        but NaN means the calculation itself is broken.
        """
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)
        result = backtest_performance_extended(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
            initial_capital=ic,
        )
        sortino = float(result.loc[result["Metric"] == "Sortino Ratio", "Value"].iloc[0])
        assert not np.isnan(sortino), (
            f"Sortino should not be NaN (may be inf for strategies with no losing days), "
            f"got {sortino}"
        )


class TestBacktestWindow:
    def test_window_restriction_changes_sharpe(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """Pre-window days must not enter the return series as zeros.

        Padding the price calendar with flat history before the backtest starts
        adds zero-return days, which shrinks Sharpe. The app passes only the
        backtest window, so metrics computed on the window must differ from
        metrics computed on a padded calendar.
        """
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)

        pad_index = pd.date_range(
            end=price_df.index[0] - pd.Timedelta(days=1), periods=750, freq="B"
        )
        padded = pd.concat(
            [pd.DataFrame(index=pad_index, columns=price_df.columns, dtype=float), price_df]
        )

        def sharpe(prices: pd.DataFrame) -> float:
            result = backtest_performance_extended(
                df_trades=df_trades,
                df_prices=prices,
                mtm_trade=mtm,
                contract_size=contract_sz,
                tons_conversion=tons_conv,
                commodity_chosen="Soybean",
                position_open=pos_open,
                initial_capital=ic,
            )
            return float(result.loc[result["Metric"] == "Sharpe Ratio", "Value"].iloc[0])

        assert abs(sharpe(padded)) < abs(sharpe(price_df))


# ---------------------------------------------------------------------------
# Summary for the first screen
# ---------------------------------------------------------------------------


class TestSummarizeBacktest:
    DATES = pd.date_range("2022-01-03", periods=10, freq="B")

    def _window(self) -> pd.Series:
        return pd.Series(100.0, index=self.DATES, name="Soybean")

    def _trades(self, positions: list[str], rows: list[int], pnl: list[float]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "trade_price": 100.0,
                "position": positions,
                "pnl_usd": pnl,
                "pnl_usd_cumsum": np.cumsum(pnl),
            },
            index=self.DATES[rows],
        )
        df.index.name = "date"
        return df

    def test_long_with_open_position(self) -> None:
        trades = self._trades(["buy", "sell", "buy"], [1, 3, 6], [0.0, 50.0, 0.0])
        mtm = {"date": self.DATES[-1], "pnl_usd": 25.0}
        summary = summarize_backtest(trades, self._window(), mtm, True)
        assert summary.position == "Long"
        assert summary.since == self.DATES[6].date()
        assert summary.round_trips == 1
        assert summary.fills == 3
        assert summary.unadjusted_pnl == 75.0

    def test_flat_after_closing(self) -> None:
        trades = self._trades(["buy", "sell"], [1, 3], [0.0, -40.0])
        summary = summarize_backtest(trades, self._window(), None, False)
        assert summary.position == "Flat"
        assert summary.since == self.DATES[3].date()
        assert summary.round_trips == 1
        assert summary.fills == 2
        assert summary.unadjusted_pnl == -40.0

    def test_empty_log_without_pnl_column(self) -> None:
        empty = pd.DataFrame(columns=["trade_price", "position", "quantity", "VaR_95"])
        summary = summarize_backtest(empty, self._window(), None, False)
        assert summary.position == "Flat"
        assert summary.since is None
        assert summary.round_trips == 0
        assert summary.fills == 0
        assert summary.time_in_market == 0.0
        assert summary.unadjusted_pnl == 0.0

    def test_time_in_market_by_hand(self) -> None:
        """Long on rows 1 and 2 (exit fill on 3), then on rows 6 to 9: 6 of 10 sessions."""
        trades = self._trades(["buy", "sell", "buy"], [1, 3, 6], [0.0, 50.0, 0.0])
        summary = summarize_backtest(trades, self._window(), None, True)
        assert summary.time_in_market == pytest.approx(0.6)

    def test_single_open_fill_has_no_round_trips(self) -> None:
        trades = self._trades(["buy"], [1], [0.0])
        summary = summarize_backtest(trades, self._window(), None, True)
        assert summary.round_trips == 0
        assert summary.fills == 1
        assert summary.time_in_market == pytest.approx(0.9)
