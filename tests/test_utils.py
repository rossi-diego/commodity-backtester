"""Unit tests for src/utils.py — PnL engine and performance metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategy import backtest
from src.utils import backtest_performance, backtest_performance_extended, pnl_trades

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
    df_trades_raw, pos_open = backtest(
        backtest_strategy="ratio",
        start_date=price_df.index[0],
        end_date=price_df.index[-1],
        df=price_df.copy(),
        up_exit=up_exit,
        down_entry=down_entry,
        commodity_chosen="Soybean",
        commodity_ratio="Corn",
        tons_conversion=tons_conv,
        contract_size=contract_sz,
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
        """Verify PnL formula against manually computed expected value."""
        dates = pd.date_range("2022-01-03", periods=4, freq="B")
        # Prices: buy at 480, sell at 520 for Soybean
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

        # Expected: (520 - 480) * 0.36744 * (136.08 * 0.36744)
        contract_tons = contract_sz * tons_conv["Soybean"]
        expected_pnl = (520.0 - 480.0) * tons_conv["Soybean"] * contract_tons
        actual_pnl = float(df_trades.loc[dates[3], "pnl_usd"])
        assert abs(actual_pnl - expected_pnl) < 0.01

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
