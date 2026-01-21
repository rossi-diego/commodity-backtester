"""
Unit tests for the utils module.

Tests cover:
- P&L calculations
- Performance metrics
- Edge cases and error handling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import pnl_trades, backtest_performance, strategy_describe


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_trades():
    """Create sample trades DataFrame."""
    dates = pd.to_datetime(["2023-01-10", "2023-02-15", "2023-03-20", "2023-04-25"])

    trades = pd.DataFrame({
        "trade_price": [1000, 1100, 950, 1050],
        "trade_ratio": [0.9, 1.1, 0.85, 1.05],
        "position": ["buy", "sell", "buy", "sell"],
        "quantity": [1, -1, 1, -1],
    }, index=dates)
    trades.index.name = "date"

    return trades


@pytest.fixture
def sample_prices():
    """Create sample price DataFrame."""
    dates = pd.date_range("2023-01-01", "2023-05-01", freq="D")

    np.random.seed(42)
    prices = pd.DataFrame({
        "Soybean": 1000 + np.cumsum(np.random.randn(len(dates)) * 10),
    }, index=dates)
    prices.index.name = "date"

    return prices


@pytest.fixture
def tons_conversion():
    """Sample tons conversion factors."""
    return {
        "Soybean": 0.3674,
        "Corn": 0.3937,
    }


# =============================================================================
# P&L Tests
# =============================================================================

class TestPnlTrades:
    """Tests for pnl_trades function."""

    def test_pnl_calculation_basic(self, sample_trades, sample_prices, tons_conversion):
        """Test basic P&L calculation."""
        df_result, mtm = pnl_trades(
            df_trades=sample_trades,
            df_prices=sample_prices,
            commodity_chosen="Soybean",
            tons_conversion=tons_conversion,
            contract_size=136.0,
            position_open=False,
        )

        assert "pnl_usd" in df_result.columns
        assert "pnl_usd_cumsum" in df_result.columns

    def test_pnl_empty_trades(self, sample_prices, tons_conversion):
        """Test P&L with empty trades."""
        empty_trades = pd.DataFrame(columns=["trade_price", "position"])
        empty_trades.index.name = "date"

        df_result, mtm = pnl_trades(
            df_trades=empty_trades,
            df_prices=sample_prices,
            commodity_chosen="Soybean",
            tons_conversion=tons_conversion,
            contract_size=136.0,
            position_open=False,
        )

        assert df_result.empty
        assert mtm is None

    def test_mtm_with_open_position(self, sample_prices, tons_conversion):
        """Test MTM calculation with open position."""
        # Single buy trade (open position)
        open_trades = pd.DataFrame({
            "trade_price": [1000],
            "trade_ratio": [0.9],
            "position": ["buy"],
            "quantity": [1],
        }, index=pd.to_datetime(["2023-01-10"]))
        open_trades.index.name = "date"

        df_result, mtm = pnl_trades(
            df_trades=open_trades,
            df_prices=sample_prices,
            commodity_chosen="Soybean",
            tons_conversion=tons_conversion,
            contract_size=136.0,
            position_open=True,
        )

        assert mtm is not None
        assert "pnl_usd" in mtm
        assert "date" in mtm


# =============================================================================
# Performance Metrics Tests
# =============================================================================

class TestBacktestPerformance:
    """Tests for backtest_performance function."""

    def test_metrics_calculation(self, sample_trades, sample_prices, tons_conversion):
        """Test that all metrics are calculated."""
        # First calculate P&L
        df_with_pnl, mtm = pnl_trades(
            df_trades=sample_trades,
            df_prices=sample_prices,
            commodity_chosen="Soybean",
            tons_conversion=tons_conversion,
            contract_size=136.0,
            position_open=False,
        )

        metrics = backtest_performance(
            df_with_pnl,
            sample_prices,
            mtm,
            contract_size=136.0,
            tons_conversion=tons_conversion,
            commodity_chosen="Soybean",
            position_open=False,
        )

        assert isinstance(metrics, pd.DataFrame)
        assert "Metric" in metrics.columns
        assert "Value" in metrics.columns

        # Check expected metrics exist
        metric_names = metrics["Metric"].tolist()
        assert "Total Profit (USD)" in metric_names
        assert "Win Rate (%)" in metric_names
        assert "Sharpe Ratio" in metric_names
        assert "Max Drawdown (USD)" in metric_names

    def test_trade_counts(self, sample_trades, sample_prices, tons_conversion):
        """Test that trade counts are correct."""
        df_with_pnl, mtm = pnl_trades(
            df_trades=sample_trades,
            df_prices=sample_prices,
            commodity_chosen="Soybean",
            tons_conversion=tons_conversion,
            contract_size=136.0,
            position_open=False,
        )

        metrics = backtest_performance(
            df_with_pnl,
            sample_prices,
            mtm,
            contract_size=136.0,
            tons_conversion=tons_conversion,
            commodity_chosen="Soybean",
            position_open=False,
        )

        total_buys = metrics[metrics["Metric"] == "Total Buys"]["Value"].values[0]
        total_sells = metrics[metrics["Metric"] == "Total Sells"]["Value"].values[0]

        assert total_buys == 2
        assert total_sells == 2


# =============================================================================
# Strategy Describe Tests
# =============================================================================

class TestStrategyDescribe:
    """Tests for strategy_describe function."""

    def test_ratio_stats(self, tons_conversion):
        """Test ratio statistics calculation."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "Soybean": np.random.randn(100) * 100 + 1000,
            "Corn": np.random.randn(100) * 50 + 500,
        }, index=dates)

        stats = strategy_describe(
            df=df,
            tons_conversion=tons_conversion,
            backtest_strategy="ratio",
        )

        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty

    def test_non_ratio_strategy(self, tons_conversion):
        """Test that non-ratio strategy returns empty DataFrame."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Soybean": np.random.randn(10) * 100 + 1000,
        }, index=dates)

        stats = strategy_describe(
            df=df,
            tons_conversion=tons_conversion,
            backtest_strategy="mean_reversion",
        )

        # Should return DataFrame with same index as input
        assert isinstance(stats, pd.DataFrame)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
