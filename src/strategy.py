"""
Backtesting engine for commodity trading strategies.

Supported strategies
--------------------
``"ratio"``        — spread ratio mean reversion
``"mean_reversion"`` — Bollinger Bands
``"momentum"``     — dual MA crossover + RSI filter
``"breakout"``     — channel breakout
``"macd"``         — MACD line crossover
"""

from __future__ import annotations

import datetime
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

StrategyType = Literal["ratio", "mean_reversion", "momentum", "breakout", "macd"]


class BacktestError(Exception):
    """Raised when backtest execution fails due to invalid inputs or data."""


def get_available_strategies() -> dict[str, str]:
    """Return a mapping of strategy key → human-readable name."""
    return {
        "ratio":          "Ratio Strategy",
        "mean_reversion": "Mean Reversion (Bollinger Bands)",
        "momentum":       "Momentum / Trend Following",
        "breakout":       "Breakout Strategy",
        "macd":           "MACD Crossover",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def backtest(
    backtest_strategy: StrategyType | str,
    start_date: datetime.date | str,
    end_date: datetime.date | str,
    df: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    # ratio params
    up_exit: float = 1.05,
    down_entry: float = 0.95,
    # mean_reversion params
    bb_window: int = 20,
    bb_std: float = 2.0,
    # momentum params
    fast_ma: int = 10,
    slow_ma: int = 30,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    # breakout params
    lookback_period: int = 20,
    breakout_threshold: float = 0.02,
    # macd params
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    # legacy compat
    window: int = 20,
) -> tuple[pd.DataFrame, bool]:
    """Run a backtest for the given strategy over the specified date range.

    Returns
    -------
    df_trades:
        Trade log with columns: trade_price, trade_ratio, position, quantity, VaR_95.
    position_open:
        True if the last signal is an unclosed long (open MTM position).
    """
    if commodity_chosen not in df.columns:
        raise BacktestError(f"Commodity '{commodity_chosen}' not found in price data.")

    df_range = df.loc[str(start_date) : str(end_date)].copy()  # type: ignore[misc]

    if df_range.empty:
        raise BacktestError(f"No price data between {start_date} and {end_date}.")

    prices = df_range[commodity_chosen]

    if backtest_strategy == "ratio":
        if not commodity_ratio or commodity_ratio not in df.columns:
            raise BacktestError(f"Ratio commodity '{commodity_ratio}' not found in data.")
        signals = _ratio_signals(df_range, commodity_chosen, commodity_ratio, tons_conversion, down_entry, up_exit)
        # Annotate full df with ratio for visualisation
        f1 = tons_conversion[commodity_chosen]
        f2 = tons_conversion[commodity_ratio]
        df["ratio"] = (df[commodity_chosen] * f1) / (df[commodity_ratio] * f2)

    elif backtest_strategy == "mean_reversion":
        signals = _bollinger_signals(prices, bb_window, bb_std)

    elif backtest_strategy == "momentum":
        signals = _momentum_signals(prices, fast_ma, slow_ma, rsi_period, rsi_oversold, rsi_overbought)

    elif backtest_strategy == "breakout":
        signals = _breakout_signals(prices, lookback_period, breakout_threshold)

    elif backtest_strategy == "macd":
        signals = _macd_signals(prices, macd_fast, macd_slow, macd_signal)

    else:
        raise BacktestError(f"Unknown strategy: '{backtest_strategy}'.")

    return _build_trades(
        signals=signals,
        prices=prices,
        df_range=df_range,
        commodity_chosen=commodity_chosen,
        commodity_ratio=commodity_ratio,
        tons_conversion=tons_conversion,
        contract_size=contract_size,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _compute_ratio(
    df: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
) -> pd.Series:
    """Return the metric-ton price ratio between two commodities."""
    return (df[commodity_chosen] * tons_conversion[commodity_chosen]) / (
        df[commodity_ratio] * tons_conversion[commodity_ratio]
    )


# ---------------------------------------------------------------------------
# Signal generators (return a Series of +1=buy, -1=sell, 0=hold)
# ---------------------------------------------------------------------------

def _ratio_signals(
    df_range: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
    down_entry: float,
    up_exit: float,
) -> pd.Series:
    f1 = tons_conversion[commodity_chosen]
    f2 = tons_conversion[commodity_ratio]
    ratio = (df_range[commodity_chosen] * f1) / (df_range[commodity_ratio] * f2)
    df_range["ratio"] = ratio
    buy_signal  = (ratio < down_entry).astype(int)
    sell_signal = (ratio > up_exit).astype(int)
    return buy_signal - sell_signal  # +1 / -1 / 0


def _bollinger_signals(prices: pd.Series, window: int, n_std: float) -> pd.Series:
    ma  = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    lower = ma - n_std * std
    upper = ma + n_std * std
    buy_signal  = (prices < lower).astype(int)
    sell_signal = (prices > upper).astype(int)
    return buy_signal - sell_signal


def _momentum_signals(
    prices: pd.Series,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    rsi_oversold: float,
    rsi_overbought: float,
) -> pd.Series:
    fast = prices.ewm(span=fast_ma, adjust=False).mean()
    slow = prices.ewm(span=slow_ma, adjust=False).mean()
    rsi  = _compute_rsi(prices, rsi_period)

    buy_signal  = ((fast > slow) & (rsi < rsi_overbought)).astype(int)
    sell_signal = ((fast < slow) | (rsi > rsi_overbought)).astype(int)
    return buy_signal - sell_signal


def _breakout_signals(prices: pd.Series, lookback: int, threshold: float) -> pd.Series:
    resistance = prices.rolling(lookback).max().shift(1)
    buy_signal  = (prices > resistance * (1 + threshold)).astype(int)
    sell_signal = (prices < resistance).astype(int)
    return buy_signal - sell_signal


def _macd_signals(prices: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast   = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow   = prices.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    buy_signal  = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    sell_signal = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    return buy_signal - sell_signal


# ---------------------------------------------------------------------------
# State machine: convert raw signals → trade log
# ---------------------------------------------------------------------------

def _build_trades(
    signals: pd.Series,
    prices: pd.Series,
    df_range: pd.DataFrame,
    commodity_chosen: str,
    commodity_ratio: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[pd.DataFrame, bool]:
    trade_type: list[str] = []
    trade_idx:  list[int] = []
    in_trade   = False
    entry_price = 0.0

    for i in range(len(signals)):
        price = float(prices.iloc[i])
        sig   = int(signals.iloc[i])

        if sig > 0 and not in_trade:
            trade_type.append("buy")
            trade_idx.append(i)
            in_trade    = True
            entry_price = price

        elif in_trade:
            # Stop loss / take profit checks
            forced_exit = False
            if stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct / 100):
                forced_exit = True
            if take_profit_pct is not None and price >= entry_price * (1 + take_profit_pct / 100):
                forced_exit = True

            if forced_exit or sig < 0:
                trade_type.append("sell")
                trade_idx.append(i)
                in_trade = False

    position_open = in_trade

    if not trade_idx:
        empty = pd.DataFrame(
            columns=["trade_price", "trade_ratio", "position", "quantity", "VaR_95"]
        )
        empty.index.name = "date"
        return empty, False

    idx_labels = df_range.index[trade_idx]
    ratio_col  = df_range.get("ratio", pd.Series(float("nan"), index=df_range.index))

    df_trades = pd.DataFrame(
        {
            "trade_price": prices.iloc[np.array(trade_idx)].values,
            "trade_ratio": ratio_col.iloc[np.array(trade_idx)].values,
            "position":    trade_type,
            "quantity":    [1 if t == "buy" else -1 for t in trade_type],
        },
        index=idx_labels,
    )
    df_trades.index.name = "date"

    # Historical VaR 95% (log-returns)
    log_returns = pd.Series(np.log(prices / prices.shift(1))).dropna()
    var_pct     = float(np.percentile(log_returns, 5))
    last_price  = float(prices.iloc[-1])
    notional    = contract_size * tons_conversion[commodity_chosen] * last_price
    df_trades["VaR_95"] = round(abs(var_pct) * notional, 2)

    return df_trades, position_open


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta  = prices.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))
