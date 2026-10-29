"""Unit tests for src/metrics.py — consistency with the headline metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import calculate_comprehensive_metrics
from src.utils import backtest_performance_extended
from tests.test_utils import _initial_capital, _run_full


def _headline(result: pd.DataFrame, metric: str) -> float:
    return float(result.loc[result["Metric"] == metric, "Value"].iloc[0])


class TestDeepAnalysisConsistency:
    def test_sharpe_and_sortino_match_headline_panel(
        self,
        price_df: pd.DataFrame,
        tons_conv: dict[str, float],
        contract_sz: float,
    ) -> None:
        """The Deep Analysis panel must report the same Sharpe/Sortino as the KPI cards.

        Previous bug: src/metrics.py annualised returns computed on trade rows
        (buy/sell records) with sqrt(252), so the same backtest showed two
        different Sharpe ratios on one screen.
        """
        df_trades, mtm, pos_open = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)

        headline = backtest_performance_extended(
            df_trades=df_trades,
            df_prices=price_df,
            mtm_trade=mtm,
            contract_size=contract_sz,
            tons_conversion=tons_conv,
            commodity_chosen="Soybean",
            position_open=pos_open,
            initial_capital=ic,
        )
        deep = calculate_comprehensive_metrics(
            df_trades=df_trades,
            df_prices=price_df,
            commodity_chosen="Soybean",
            initial_capital=ic,
            mtm_trade=mtm,
        )

        # The headline table rounds to 4 decimals for display.
        np.testing.assert_allclose(
            deep.sharpe_ratio, _headline(headline, "Sharpe Ratio"), atol=1e-4
        )
        np.testing.assert_allclose(
            deep.sortino_ratio, _headline(headline, "Sortino Ratio"), atol=1e-4
        )

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
        df_trades, mtm, _ = _run_full(price_df, tons_conv, contract_sz)
        if df_trades.empty:
            pytest.skip("No trades generated.")
        ic = _initial_capital(price_df, tons_conv, contract_sz)

        pad_index = pd.date_range(
            end=price_df.index[0] - pd.Timedelta(days=1), periods=750, freq="B"
        )
        padded = pd.concat([pd.DataFrame(index=pad_index, columns=price_df.columns), price_df])

        on_window = calculate_comprehensive_metrics(
            df_trades, price_df, "Soybean", initial_capital=ic, mtm_trade=mtm
        )
        on_padded = calculate_comprehensive_metrics(
            df_trades, padded, "Soybean", initial_capital=ic, mtm_trade=mtm
        )
        assert abs(on_padded.sharpe_ratio) < abs(on_window.sharpe_ratio)
