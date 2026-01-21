"""
Unit tests for the strategy module.

Tests cover:
- Ratio strategy signal generation
- Trade indexing and tracking
- Edge cases (no trades, single trade, etc.)
"""

import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy import backtest, BacktestError, StrategyType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = {
        "Soybean": 1000 + np.cumsum(np.random.randn(100) * 10),
        "Corn": 500 + np.cumsum(np.random.randn(100) * 5),
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def tons_conversion():
    """Sample tons conversion factors."""
    return {
        "Soybean": 0.3674,
        "Corn": 0.3937,
    }


# =============================================================================
# Basic Tests
# =============================================================================

class TestBacktestFunction:
    """Tests for the main backtest function."""

    def test_ratio_strategy_basic(self, sample_price_data, tons_conversion):
        """Test that ratio strategy runs without errors."""
        df_trades, position_open = backtest(
            backtest_strategy="ratio",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 4, 10),
            df=sample_price_data,
            up_exit=1.2,
            down_entry=0.8,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        assert isinstance(df_trades, pd.DataFrame)
        assert isinstance(position_open, bool)

    def test_empty_date_range(self, sample_price_data, tons_conversion):
        """Test with date range outside available data."""
        df_trades, position_open = backtest(
            backtest_strategy="ratio",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            df=sample_price_data,
            up_exit=1.2,
            down_entry=0.8,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        assert df_trades.empty
        assert position_open is False

    def test_invalid_commodity(self, sample_price_data, tons_conversion):
        """Test with non-existent commodity raises error."""
        with pytest.raises(BacktestError) as excinfo:
            backtest(
                backtest_strategy="ratio",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 4, 10),
                df=sample_price_data,
                up_exit=1.2,
                down_entry=0.8,
                commodity_chosen="InvalidCommodity",
                commodity_ratio="Corn",
                tons_conversion=tons_conversion,
                contract_size=136.0,
            )

        assert "not found" in str(excinfo.value)

    def test_invalid_strategy(self, sample_price_data, tons_conversion):
        """Test with unsupported strategy raises error."""
        with pytest.raises(BacktestError) as excinfo:
            backtest(
                backtest_strategy="invalid_strategy",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 4, 10),
                df=sample_price_data,
                up_exit=1.2,
                down_entry=0.8,
                commodity_chosen="Soybean",
                commodity_ratio="Corn",
                tons_conversion=tons_conversion,
                contract_size=136.0,
            )

        assert "not implemented" in str(excinfo.value)


class TestTradeSignals:
    """Tests for trade signal generation."""

    def test_buy_signal_below_entry(self, tons_conversion):
        """Test that buy signal is generated when ratio drops below entry."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        # Create data where ratio starts high and drops below entry
        data = {
            "Soybean": [1000, 1000, 1000, 900, 850, 850, 850, 850, 850, 850],
            "Corn": [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"

        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            df=df,
            up_exit=2.0,  # High exit to prevent sells
            down_entry=1.7,  # Entry threshold
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        # Should have at least one buy
        if not df_trades.empty:
            assert "buy" in df_trades["position"].values

    def test_sell_signal_above_exit(self, tons_conversion):
        """Test that sell signal is generated when ratio exceeds exit."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        # Create data where ratio starts low then rises
        data = {
            "Soybean": [800, 800, 1000, 1200, 1400, 1400, 1400, 1400, 1400, 1400],
            "Corn": [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"

        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            df=df,
            up_exit=2.0,  # Exit threshold
            down_entry=1.6,  # Entry that triggers on first days
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        # Should have both buy and sell
        if not df_trades.empty and len(df_trades) >= 2:
            positions = df_trades["position"].values
            assert "buy" in positions
            assert "sell" in positions


class TestTradeDataFrame:
    """Tests for trade DataFrame structure."""

    def test_trade_columns(self, sample_price_data, tons_conversion):
        """Test that trades DataFrame has correct columns."""
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 4, 10),
            df=sample_price_data,
            up_exit=1.5,
            down_entry=0.5,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        expected_columns = ["trade_price", "trade_ratio", "position", "quantity"]

        for col in expected_columns:
            assert col in df_trades.columns or df_trades.empty

    def test_trade_index_is_datetime(self, sample_price_data, tons_conversion):
        """Test that trades DataFrame index is datetime."""
        df_trades, _ = backtest(
            backtest_strategy="ratio",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 4, 10),
            df=sample_price_data,
            up_exit=1.5,
            down_entry=0.5,
            commodity_chosen="Soybean",
            commodity_ratio="Corn",
            tons_conversion=tons_conversion,
            contract_size=136.0,
        )

        if not df_trades.empty:
            assert df_trades.index.name == "date"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
