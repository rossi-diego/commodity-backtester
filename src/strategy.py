"""
Backtesting strategy module.

This module implements various trading strategies for commodity backtesting:
- Ratio Strategy: Trade based on price ratios between commodities
- Mean Reversion: Bollinger Bands based mean reversion
- Momentum: Trend following with MA crossover and RSI
- Breakout: Support/resistance breakout strategy

Each strategy can be configured with specific parameters and generates
trade signals based on technical indicators.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class StrategyType(Enum):
    """Available backtesting strategy types."""
    RATIO = "ratio"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    MACD = "macd"


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
        signal_value: Strategy-specific signal value (e.g., RSI, z-score)
    """
    date: pd.Timestamp
    price: float
    ratio: Optional[float]
    position: str
    quantity: int
    signal_value: Optional[float] = None


@dataclass
class StrategyConfig:
    """
    Configuration parameters for trading strategies.

    Attributes:
        strategy_type: Type of strategy to execute
        commodity_chosen: Primary commodity to trade
        commodity_ratio: Secondary commodity for ratio calculation
        tons_conversion: Dictionary of tons conversion factors
        contract_size: Size of one contract

        # Ratio strategy parameters
        up_exit: Upper threshold for ratio exit signal
        down_entry: Lower threshold for ratio entry signal

        # Mean reversion parameters
        bb_window: Bollinger Bands window period
        bb_std: Number of standard deviations for bands

        # Momentum parameters
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period
        rsi_period: RSI calculation period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold

        # Breakout parameters
        lookback_period: Period for support/resistance calculation
        breakout_threshold: Percentage above/below for breakout confirmation
        volume_confirmation: Whether to require volume confirmation
    """
    strategy_type: str
    commodity_chosen: str
    commodity_ratio: str = ""
    tons_conversion: Dict[str, float] = field(default_factory=dict)
    contract_size: float = 136.0

    # Ratio strategy
    up_exit: float = 1.05
    down_entry: float = 0.95

    # Mean reversion (Bollinger Bands)
    bb_window: int = 20
    bb_std: float = 2.0

    # Momentum
    fast_ma: int = 10
    slow_ma: int = 30
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Breakout
    lookback_period: int = 20
    breakout_threshold: float = 0.02
    volume_confirmation: bool = False

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Risk management parameters
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


class BacktestError(Exception):
    """Custom exception for backtesting errors."""
    pass


# =============================================================================
# Technical Indicators
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of prices
        period: RSI calculation period

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of prices
        window: Moving average window
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return middle, upper, lower


def calculate_moving_averages(
    prices: pd.Series,
    fast_period: int = 10,
    slow_period: int = 30
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate fast and slow moving averages.

    Args:
        prices: Series of prices
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        Tuple of (fast_ma, slow_ma)
    """
    fast_ma = prices.rolling(window=fast_period).mean()
    slow_ma = prices.rolling(window=slow_period).mean()

    return fast_ma, slow_ma


def calculate_support_resistance(
    prices: pd.Series,
    lookback: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate rolling support and resistance levels.

    Args:
        prices: Series of prices
        lookback: Lookback period

    Returns:
        Tuple of (support, resistance)
    """
    support = prices.rolling(window=lookback).min()
    resistance = prices.rolling(window=lookback).max()

    return support, resistance


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        Series of ATR values
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =============================================================================
# Strategy Implementations
# =============================================================================

def _execute_ratio_strategy(
    df_filtered: pd.DataFrame,
    df_full: pd.DataFrame,
    config: StrategyConfig
) -> Tuple[List[Dict], bool, pd.DataFrame]:
    """
    Execute ratio-based trading strategy.

    Buys when the ratio drops below entry threshold and sells when
    it rises above exit threshold.

    Args:
        df_filtered: Price data filtered to backtest period
        df_full: Full price dataset
        config: Strategy configuration

    Returns:
        Tuple of (trades list, position open flag, updated DataFrame)
    """
    commodity_chosen = config.commodity_chosen
    commodity_ratio = config.commodity_ratio

    factor_1 = config.tons_conversion[commodity_chosen]
    factor_2 = config.tons_conversion[commodity_ratio]

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
        if ratio <= config.down_entry and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": ratio,
                "position": "buy",
                "quantity": 1,
                "signal_value": ratio
            })
            position_open = True
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, ratio={ratio:.4f}")

        # Exit signal: ratio above threshold and position is open
        elif ratio > config.up_exit and position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": ratio,
                "position": "sell",
                "quantity": -1,
                "signal_value": ratio
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
    config: StrategyConfig
) -> Tuple[List[Dict], bool]:
    """
    Execute mean reversion strategy using Bollinger Bands.

    Buys at the lower band and sells at the upper band.

    Args:
        df_filtered: Price data filtered to backtest period
        config: Strategy configuration

    Returns:
        Tuple of (trades list, position open flag)
    """
    commodity = config.commodity_chosen
    prices = df_filtered[commodity]

    # Calculate Bollinger Bands
    middle, upper, lower = calculate_bollinger_bands(
        prices,
        window=config.bb_window,
        num_std=config.bb_std
    )

    df_filtered["bb_middle"] = middle
    df_filtered["bb_upper"] = upper
    df_filtered["bb_lower"] = lower

    # Calculate z-score for signal value
    df_filtered["z_score"] = (prices - middle) / (upper - middle) * config.bb_std

    trades: List[Dict] = []
    position_open = False

    for idx, row in df_filtered.iterrows():
        price = row[commodity]
        upper_band = row["bb_upper"]
        lower_band = row["bb_lower"]
        z_score = row["z_score"]

        if pd.isna(upper_band) or pd.isna(lower_band):
            continue

        # Entry signal: price at lower band
        if price <= lower_band and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "buy",
                "quantity": 1,
                "signal_value": z_score
            })
            position_open = True
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, z_score={z_score:.2f}")

        # Exit signal: price at upper band
        elif price >= upper_band and position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "sell",
                "quantity": -1,
                "signal_value": z_score
            })
            position_open = False
            logger.debug(f"SELL signal at {idx}: price={price:.2f}, z_score={z_score:.2f}")

    return trades, position_open


def _execute_momentum_strategy(
    df_filtered: pd.DataFrame,
    config: StrategyConfig
) -> Tuple[List[Dict], bool]:
    """
    Execute momentum/trend following strategy.

    Uses MA crossover combined with RSI filter:
    - BUY: Fast MA crosses above Slow MA AND RSI > oversold
    - SELL: Fast MA crosses below Slow MA OR RSI > overbought

    Args:
        df_filtered: Price data filtered to backtest period
        config: Strategy configuration

    Returns:
        Tuple of (trades list, position open flag)
    """
    commodity = config.commodity_chosen
    prices = df_filtered[commodity]

    # Calculate indicators
    fast_ma, slow_ma = calculate_moving_averages(
        prices,
        fast_period=config.fast_ma,
        slow_period=config.slow_ma
    )
    rsi = calculate_rsi(prices, period=config.rsi_period)

    df_filtered["fast_ma"] = fast_ma
    df_filtered["slow_ma"] = slow_ma
    df_filtered["rsi"] = rsi

    # Generate crossover signals
    df_filtered["ma_diff"] = fast_ma - slow_ma
    df_filtered["ma_diff_prev"] = df_filtered["ma_diff"].shift(1)

    trades: List[Dict] = []
    position_open = False

    for idx, row in df_filtered.iterrows():
        price = row[commodity]
        ma_diff = row["ma_diff"]
        ma_diff_prev = row["ma_diff_prev"]
        current_rsi = row["rsi"]

        if pd.isna(ma_diff) or pd.isna(ma_diff_prev):
            continue

        # Bullish crossover: fast MA crosses above slow MA
        bullish_crossover = (ma_diff > 0) and (ma_diff_prev <= 0)
        # Bearish crossover: fast MA crosses below slow MA
        bearish_crossover = (ma_diff < 0) and (ma_diff_prev >= 0)

        # Entry signal: bullish crossover with RSI confirmation
        if bullish_crossover and current_rsi > config.rsi_oversold and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "buy",
                "quantity": 1,
                "signal_value": current_rsi
            })
            position_open = True
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, RSI={current_rsi:.1f}")

        # Exit signal: bearish crossover OR RSI overbought
        elif position_open and (bearish_crossover or current_rsi > config.rsi_overbought):
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "sell",
                "quantity": -1,
                "signal_value": current_rsi
            })
            position_open = False
            logger.debug(f"SELL signal at {idx}: price={price:.2f}, RSI={current_rsi:.1f}")

    return trades, position_open


def _execute_breakout_strategy(
    df_filtered: pd.DataFrame,
    config: StrategyConfig
) -> Tuple[List[Dict], bool]:
    """
    Execute breakout strategy based on support/resistance levels.

    - BUY: Price breaks above resistance with confirmation
    - SELL: Price breaks below support OR hits trailing stop

    Args:
        df_filtered: Price data filtered to backtest period
        config: Strategy configuration

    Returns:
        Tuple of (trades list, position open flag)
    """
    commodity = config.commodity_chosen
    prices = df_filtered[commodity]

    # Calculate support and resistance
    support, resistance = calculate_support_resistance(
        prices,
        lookback=config.lookback_period
    )

    df_filtered["support"] = support
    df_filtered["resistance"] = resistance

    # Calculate breakout thresholds
    df_filtered["breakout_up"] = resistance * (1 + config.breakout_threshold)
    df_filtered["breakout_down"] = support * (1 - config.breakout_threshold)

    trades: List[Dict] = []
    position_open = False
    entry_price = 0.0
    highest_since_entry = 0.0

    for idx, row in df_filtered.iterrows():
        price = row[commodity]
        res = row["resistance"]
        sup = row["support"]
        breakout_up = row["breakout_up"]
        breakout_down = row["breakout_down"]

        if pd.isna(res) or pd.isna(sup):
            continue

        # Track highest price since entry for trailing stop
        if position_open:
            highest_since_entry = max(highest_since_entry, price)

        # Entry signal: breakout above resistance
        if price > breakout_up and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "buy",
                "quantity": 1,
                "signal_value": price / res - 1  # Breakout percentage
            })
            position_open = True
            entry_price = price
            highest_since_entry = price
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, resistance={res:.2f}")

        # Exit signal: breakdown below support OR trailing stop
        elif position_open:
            # Trailing stop: 5% below highest price since entry
            trailing_stop = highest_since_entry * 0.95

            if price < breakout_down or price < trailing_stop:
                trades.append({
                    "date": idx,
                    "trade_price": price,
                    "trade_ratio": np.nan,
                    "position": "sell",
                    "quantity": -1,
                    "signal_value": price / entry_price - 1  # Return since entry
                })
                position_open = False
                logger.debug(f"SELL signal at {idx}: price={price:.2f}, support={sup:.2f}")

    return trades, position_open


def _execute_macd_strategy(
    df_filtered: pd.DataFrame,
    config: StrategyConfig
) -> Tuple[List[Dict], bool]:
    """
    Execute MACD-based trading strategy.

    Uses MACD crossovers for entry/exit:
    - BUY: MACD line crosses above signal line (bullish crossover)
    - SELL: MACD line crosses below signal line (bearish crossover)

    Args:
        df_filtered: Price data filtered to backtest period
        config: Strategy configuration

    Returns:
        Tuple of (trades list, position open flag)
    """
    commodity = config.commodity_chosen
    prices = df_filtered[commodity]

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(
        prices,
        fast_period=config.macd_fast,
        slow_period=config.macd_slow,
        signal_period=config.macd_signal
    )

    df_filtered["macd"] = macd_line
    df_filtered["macd_signal"] = signal_line
    df_filtered["macd_histogram"] = histogram

    trades: List[Dict] = []
    position_open = False
    entry_price = 0.0

    for idx, row in df_filtered.iterrows():
        price = row[commodity]
        macd = row["macd"]
        signal = row["macd_signal"]

        if pd.isna(macd) or pd.isna(signal):
            continue

        # Get previous values for crossover detection
        prev_idx = df_filtered.index.get_loc(idx)
        if prev_idx == 0:
            continue

        prev_row = df_filtered.iloc[prev_idx - 1]
        prev_macd = prev_row["macd"]
        prev_signal = prev_row["macd_signal"]

        if pd.isna(prev_macd) or pd.isna(prev_signal):
            continue

        # Detect crossovers
        bullish_crossover = (prev_macd <= prev_signal) and (macd > signal)
        bearish_crossover = (prev_macd >= prev_signal) and (macd < signal)

        # Risk management checks
        if position_open and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * 100

            # Stop loss check
            if config.stop_loss_pct and pnl_pct <= -config.stop_loss_pct:
                trades.append({
                    "date": idx,
                    "trade_price": price,
                    "trade_ratio": np.nan,
                    "position": "sell",
                    "quantity": -1,
                    "signal_value": macd - signal
                })
                position_open = False
                entry_price = 0.0
                logger.debug(f"STOP LOSS at {idx}: price={price:.2f}, loss={pnl_pct:.2f}%")
                continue

            # Take profit check
            if config.take_profit_pct and pnl_pct >= config.take_profit_pct:
                trades.append({
                    "date": idx,
                    "trade_price": price,
                    "trade_ratio": np.nan,
                    "position": "sell",
                    "quantity": -1,
                    "signal_value": macd - signal
                })
                position_open = False
                entry_price = 0.0
                logger.debug(f"TAKE PROFIT at {idx}: price={price:.2f}, gain={pnl_pct:.2f}%")
                continue

        # Entry signal: bullish MACD crossover
        if bullish_crossover and not position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "buy",
                "quantity": 1,
                "signal_value": macd - signal
            })
            position_open = True
            entry_price = price
            logger.debug(f"BUY signal at {idx}: price={price:.2f}, MACD={macd:.4f}")

        # Exit signal: bearish MACD crossover
        elif bearish_crossover and position_open:
            trades.append({
                "date": idx,
                "trade_price": price,
                "trade_ratio": np.nan,
                "position": "sell",
                "quantity": -1,
                "signal_value": macd - signal
            })
            position_open = False
            entry_price = 0.0
            logger.debug(f"SELL signal at {idx}: price={price:.2f}, MACD={macd:.4f}")

    return trades, position_open


# =============================================================================
# Main Backtest Function
# =============================================================================

def backtest(
    backtest_strategy: str,
    start_date: date,
    end_date: date,
    df: pd.DataFrame,
    up_exit: float = 1.05,
    down_entry: float = 0.95,
    commodity_chosen: str = "",
    commodity_ratio: str = "",
    tons_conversion: Optional[Dict[str, float]] = None,
    contract_size: float = 136.0,
    window: int = 20,
    **kwargs
) -> Tuple[pd.DataFrame, bool]:
    """
    Execute a backtest simulation using the specified strategy.

    Simulates trading based on entry and exit signals, tracking all trades
    and calculating associated risk metrics.

    Args:
        backtest_strategy: Strategy type ('ratio', 'mean_reversion', 'momentum', 'breakout')
        start_date: Backtest start date
        end_date: Backtest end date
        df: DataFrame containing commodity price data
        up_exit: Upper threshold to trigger exit signal (ratio strategy)
        down_entry: Lower threshold to trigger entry signal (ratio strategy)
        commodity_chosen: Primary commodity to trade
        commodity_ratio: Secondary commodity for ratio calculation
        tons_conversion: Dictionary of tons conversion factors
        contract_size: Size of one contract
        window: Rolling window for strategies (default: 20)
        **kwargs: Additional strategy-specific parameters

    Returns:
        Tuple containing:
            - DataFrame with all executed trades
            - Boolean indicating if a position is still open

    Raises:
        BacktestError: If strategy is not implemented or validation fails

    Example:
        >>> trades_df, is_open = backtest(
        ...     backtest_strategy="momentum",
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2023, 12, 31),
        ...     df=price_data,
        ...     commodity_chosen="Soybean",
        ...     fast_ma=10,
        ...     slow_ma=30,
        ...     rsi_period=14
        ... )
    """
    tons_conversion = tons_conversion or {}

    # Validate inputs
    if commodity_chosen not in df.columns:
        raise BacktestError(f"Commodity '{commodity_chosen}' not found in data")

    if backtest_strategy == StrategyType.RATIO.value:
        if commodity_ratio not in df.columns:
            raise BacktestError(f"Ratio commodity '{commodity_ratio}' not found in data")
        if commodity_chosen not in tons_conversion:
            raise BacktestError(f"Tons conversion not found for '{commodity_chosen}'")
        if commodity_ratio not in tons_conversion:
            raise BacktestError(f"Tons conversion not found for '{commodity_ratio}'")

    # Filter data to backtest period
    df_filtered = df.loc[str(start_date):str(end_date)].copy()

    if df_filtered.empty:
        logger.warning("No data available for the specified date range")
        return pd.DataFrame(), False

    # Build strategy configuration
    config = StrategyConfig(
        strategy_type=backtest_strategy,
        commodity_chosen=commodity_chosen,
        commodity_ratio=commodity_ratio,
        tons_conversion=tons_conversion,
        contract_size=contract_size,
        up_exit=up_exit,
        down_entry=down_entry,
        bb_window=kwargs.get("bb_window", window),
        bb_std=kwargs.get("bb_std", 2.0),
        fast_ma=kwargs.get("fast_ma", 10),
        slow_ma=kwargs.get("slow_ma", 30),
        rsi_period=kwargs.get("rsi_period", 14),
        rsi_overbought=kwargs.get("rsi_overbought", 70.0),
        rsi_oversold=kwargs.get("rsi_oversold", 30.0),
        lookback_period=kwargs.get("lookback_period", window),
        breakout_threshold=kwargs.get("breakout_threshold", 0.02),
        volume_confirmation=kwargs.get("volume_confirmation", False),
        macd_fast=kwargs.get("macd_fast", 12),
        macd_slow=kwargs.get("macd_slow", 26),
        macd_signal=kwargs.get("macd_signal", 9),
        stop_loss_pct=kwargs.get("stop_loss_pct"),
        take_profit_pct=kwargs.get("take_profit_pct"),
    )

    trades: List[Dict] = []
    position_open = False

    # Execute appropriate strategy
    if backtest_strategy == StrategyType.RATIO.value:
        trades, position_open, df_filtered = _execute_ratio_strategy(
            df_filtered=df_filtered,
            df_full=df,
            config=config
        )

    elif backtest_strategy == StrategyType.MEAN_REVERSION.value:
        trades, position_open = _execute_mean_reversion_strategy(
            df_filtered=df_filtered,
            config=config
        )

    elif backtest_strategy == StrategyType.MOMENTUM.value:
        trades, position_open = _execute_momentum_strategy(
            df_filtered=df_filtered,
            config=config
        )

    elif backtest_strategy == StrategyType.BREAKOUT.value:
        trades, position_open = _execute_breakout_strategy(
            df_filtered=df_filtered,
            config=config
        )

    elif backtest_strategy == StrategyType.MACD.value:
        trades, position_open = _execute_macd_strategy(
            df_filtered=df_filtered,
            config=config
        )

    else:
        raise BacktestError(f"Strategy '{backtest_strategy}' is not implemented")

    # Create trades DataFrame with proper indexing
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades.set_index("date", inplace=True)
        df_trades.index.name = "date"
    else:
        df_trades = pd.DataFrame(
            columns=["trade_price", "trade_ratio", "position", "quantity", "signal_value"]
        )
        df_trades.index.name = "date"

    # Calculate VaR (95%) for risk metrics
    if not df_trades.empty and not df_filtered.empty:
        df_filtered["returns"] = df_filtered[commodity_chosen].pct_change()
        var_95 = df_filtered["returns"].dropna().quantile(0.05)
        last_price = df_filtered[commodity_chosen].iloc[-1]

        tons_factor = tons_conversion.get(commodity_chosen, 1.0)
        df_trades["VaR_95"] = var_95 * contract_size * tons_factor * last_price

    logger.info(
        f"Backtest completed: strategy={backtest_strategy}, "
        f"trades={len(trades)}, position_open={position_open}"
    )

    return df_trades, position_open


def get_strategy_parameters(strategy_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a given strategy type.

    Args:
        strategy_type: Strategy type string

    Returns:
        Dictionary of parameter names and default values
    """
    common_params = {
        "commodity_chosen": "Soybean",
        "contract_size": 136.0,
    }

    strategy_params = {
        StrategyType.RATIO.value: {
            **common_params,
            "commodity_ratio": "Corn",
            "up_exit": 1.05,
            "down_entry": 0.95,
        },
        StrategyType.MEAN_REVERSION.value: {
            **common_params,
            "bb_window": 20,
            "bb_std": 2.0,
        },
        StrategyType.MOMENTUM.value: {
            **common_params,
            "fast_ma": 10,
            "slow_ma": 30,
            "rsi_period": 14,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
        },
        StrategyType.BREAKOUT.value: {
            **common_params,
            "lookback_period": 20,
            "breakout_threshold": 0.02,
        },
        StrategyType.MACD.value: {
            **common_params,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
    }

    return strategy_params.get(strategy_type, common_params)


def get_available_strategies() -> List[Dict[str, str]]:
    """
    Get list of available strategies with descriptions.

    Returns:
        List of dictionaries with strategy info
    """
    return [
        {
            "type": StrategyType.RATIO.value,
            "name": "Ratio Trading",
            "description": "Trade based on price ratios between two commodities in metric ton terms"
        },
        {
            "type": StrategyType.MEAN_REVERSION.value,
            "name": "Mean Reversion (Bollinger Bands)",
            "description": "Buy at lower band, sell at upper band - assumes prices revert to mean"
        },
        {
            "type": StrategyType.MOMENTUM.value,
            "name": "Momentum / Trend Following",
            "description": "MA crossover with RSI filter - follows established trends"
        },
        {
            "type": StrategyType.BREAKOUT.value,
            "name": "Breakout",
            "description": "Enter on resistance breakout, exit on support breakdown with trailing stop"
        },
        {
            "type": StrategyType.MACD.value,
            "name": "MACD Crossover",
            "description": "Uses MACD line crossovers with signal line for entry/exit - good for trending markets"
        },
    ]
