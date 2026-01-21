"""
Backtesting strategy module.

This module implements various trading strategies for commodity backtesting,
including ratio-based trading and mean reversion strategies.
"""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available backtesting strategy types."""
    RATIO = "ratio"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class Trade:
    """
    Represents a single trade execution.

    Attributes:
        date: Trade execution date
        price: Execution price
        ratio: Price ratio at execution (for ratio strategy)
        position: Trade direction ('buy' or 'sell')
        quantity: Number of contracts (positive for buy, negative for sell)
    """
    date: pd.Timestamp
    price: float
    ratio: Optional[float]
    position: str
    quantity: int


class BacktestError(Exception):
    """Custom exception for backtesting errors."""
    pass


def backtest(
    backtest_strategy: str,
    start_date: date,
    end_date: date,
    df: pd.DataFrame,
    up_exit: float,
    down_entry: float,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: Dict[str, float],
    contract_size: float,
    window: int = 20
) -> Tuple[pd.DataFrame, bool]:
    """
    Execute a backtest simulation using the specified strategy.

    Simulates trading based on entry and exit signals, tracking all trades
    and calculating associated risk metrics.

    Args:
        backtest_strategy: Strategy type ('ratio' or 'mean_reversion')
        start_date: Backtest start date
        end_date: Backtest end date
        df: DataFrame containing commodity price data
        up_exit: Upper threshold to trigger exit signal
        down_entry: Lower threshold to trigger entry signal
        commodity_chosen: Primary commodity to trade
        commodity_ratio: Secondary commodity for ratio calculation
        tons_conversion: Dictionary of tons conversion factors
        contract_size: Size of one contract
        window: Rolling window for mean reversion (default: 20)

    Returns:
        Tuple containing:
            - DataFrame with all executed trades
            - Boolean indicating if a position is still open

    Raises:
        BacktestError: If strategy is not implemented or validation fails

    Example:
        >>> trades_df, is_open = backtest(
        ...     backtest_strategy="ratio",
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2023, 12, 31),
        ...     df=price_data,
        ...     up_exit=1.05,
        ...     down_entry=0.95,
        ...     commodity_chosen="Soybean",
        ...     commodity_ratio="Corn",
        ...     tons_conversion=tons_conv,
        ...     contract_size=136.0
        ... )
    """
    # Validate inputs
    if commodity_chosen not in df.columns:
        raise BacktestError(f"Commodity '{commodity_chosen}' not found in data")
    if commodity_ratio not in df.columns:
        raise BacktestError(f"Ratio commodity '{commodity_ratio}' not found in data")

    # Filter data to backtest period
    df_filtered = df.loc[str(start_date):str(end_date)].copy()

    if df_filtered.empty:
        logger.warning("No data available for the specified date range")
        return pd.DataFrame(), False

    trades: List[Dict] = []
    position_open = False

    if backtest_strategy == StrategyType.RATIO.value:
        trades, position_open, df_filtered = _execute_ratio_strategy(
            df_filtered=df_filtered,
            df_full=df,
            commodity_chosen=commodity_chosen,
            commodity_ratio=commodity_ratio,
            tons_conversion=tons_conversion,
            down_entry=down_entry,
            up_exit=up_exit
        )

    elif backtest_strategy == StrategyType.MEAN_REVERSION.value:
        trades, position_open = _execute_mean_reversion_strategy(
            df_filtered=df_filtered,
            commodity_chosen=commodity_chosen,
            window=window
        )

    else:
        raise BacktestError(f"Strategy '{backtest_strategy}' is not implemented")

    # Create trades DataFrame with proper indexing
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades.set_index("date", inplace=True)
        df_trades.index.name = "date"
    else:
        df_trades = pd.DataFrame(columns=["trade_price", "trade_ratio", "position", "quantity"])
        df_trades.index.name = "date"

    # Calculate VaR (95%) for risk metrics
    if not df_trades.empty and not df_filtered.empty:
        df_filtered["returns"] = df_filtered[commodity_chosen].pct_change()
        var_95 = df_filtered["returns"].dropna().quantile(0.05)
        last_price = df_filtered[commodity_chosen].iloc[-1]
        df_trades["VaR_95"] = var_95 * contract_size * tons_conversion[commodity_chosen] * last_price

    logger.info(f"Backtest completed: {len(trades)} trades, position_open={position_open}")

    return df_trades, position_open


def _execute_ratio_strategy(
    df_filtered: pd.DataFrame,
    df_full: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: Dict[str, float],
    down_entry: float,
    up_exit: float
) -> Tuple[List[Dict], bool, pd.DataFrame]:
    """
    Execute ratio-based trading strategy.

    Buys when the ratio drops below entry threshold and sells when
    it rises above exit threshold.

    Args:
        df_filtered: Price data filtered to backtest period
        df_full: Full price dataset
        commodity_chosen: Primary commodity
        commodity_ratio: Secondary commodity for ratio
        tons_conversion: Conversion factors
        down_entry: Entry threshold (buy signal)
        up_exit: Exit threshold (sell signal)

    Returns:
        Tuple of (trades list, position open flag, updated DataFrame)
    """
    factor_1 = tons_conversion[commodity_chosen]
    factor_2 = tons_conversion[commodity_ratio]

    # Calculate price ratio (in metric ton terms)
    df_filtered["ratio"] = (
        (df_filtered[commodity_chosen] * factor_1) /
        (df_filtered[commodity_ratio] * factor_2)
    )

    trades: List[Dict] = []
    position_open = False

    for idx, row in df_filtered.iterrows():
        price = row[commodity_chosen]
        ratio = row["ratio"]

        # Entry signal: ratio below threshold and no open position
        if ratio <= down_entry and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": ratio,
                "position": "buy",
                "quantity": 1
            })
            position_open = True
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, ratio={ratio:.4f}")

        # Exit signal: ratio above threshold and position is open
        elif ratio > up_exit and position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": ratio,
                "position": "sell",
                "quantity": -1
            })
            position_open = False
            logger.debug(f"SELL signal at {idx}: price={price:.2f}, ratio={ratio:.4f}")

    # Update full DataFrame with ratio column for visualization
    df_full["ratio"] = (
        (df_full[commodity_chosen] * factor_1) /
        (df_full[commodity_ratio] * factor_2)
    )

    return trades, position_open, df_filtered


def _execute_mean_reversion_strategy(
    df_filtered: pd.DataFrame,
    commodity_chosen: str,
    window: int = 20
) -> Tuple[List[Dict], bool]:
    """
    Execute mean reversion strategy using Bollinger Bands.

    Buys at the lower band and sells at the upper band.

    Args:
        df_filtered: Price data filtered to backtest period
        commodity_chosen: Commodity to trade
        window: Rolling window for moving average (default: 20)

    Returns:
        Tuple of (trades list, position open flag)
    """
    # Calculate Bollinger Bands
    df_filtered["moving_average"] = df_filtered[commodity_chosen].rolling(window=window).mean()
    df_filtered["std_dev"] = df_filtered[commodity_chosen].rolling(window=window).std()

    trades: List[Dict] = []
    position_open = False

    for idx, row in df_filtered.iterrows():
        price = row[commodity_chosen]
        ma = row["moving_average"]
        std = row["std_dev"]

        if pd.isna(ma) or pd.isna(std):
            continue

        upper_band = ma + 2 * std
        lower_band = ma - 2 * std

        # Entry signal: price at lower band
        if price <= lower_band and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "buy",
                "quantity": 1
            })
            position_open = True

        # Exit signal: price at upper band
        elif price >= upper_band and position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "sell",
                "quantity": -1
            })
            position_open = False

    return trades, position_open
