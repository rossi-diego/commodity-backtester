"""
Backtesting engine for commodity trading rules.

Supported rules
---------------
``"tsmom_12m"`` - 12-month time-series momentum, checked at each month end
``"donchian"``  - Donchian channel breakout (20-session entry, 10-session exit)
``"buy_hold"``  - reference: long from the first fill to the window end

Execution model
---------------
Signals are generated from the **close of day T** and executed at the
**close of day T+1** (next-bar execution).  This avoids the look-ahead
bias of same-bar execution, where the closing price that triggers the
signal is simultaneously used as the fill price.

Signals run on the instrument's own history up to the window end and are then
sliced to the window. Using the instrument's own series keeps holidays of other
markets out of the rolling windows, and using the history before the window
means a short window does not start blind.
"""

from __future__ import annotations

import datetime
from typing import Literal

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

StrategyType = Literal["tsmom_12m", "donchian", "buy_hold"]

__all__ = [
    "BacktestError",
    "StrategyType",
    "backtest",
]


class BacktestError(Exception):
    """Raised when backtest execution fails due to invalid inputs or data."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def backtest(
    backtest_strategy: StrategyType | str,
    start_date: datetime.date | str,
    end_date: datetime.date | str,
    df: pd.DataFrame,
    commodity_chosen: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    tsmom_lookback: int = 252,
    donchian_entry: int = 20,
    donchian_exit: int = 10,
) -> tuple[pd.DataFrame, bool]:
    """Run a backtest for the given rule over the specified date range.

    Returns
    -------
    df_trades:
        Trade log with columns: trade_price, position, quantity, VaR_95.  Trade
        prices reflect **next-bar (T+1) execution**: the signal fires on day T's
        close and the fill is recorded at day T+1's close.
    position_open:
        True if the last signal is an unclosed long (open MTM position).
    """
    if commodity_chosen not in df.columns:
        raise BacktestError(f"Commodity '{commodity_chosen}' not found in price data.")

    history = df[commodity_chosen].loc[: str(end_date)].dropna()

    if backtest_strategy == "tsmom_12m":
        signals_full = _tsmom_signals(history, tsmom_lookback)
    elif backtest_strategy == "donchian":
        signals_full = _donchian_signals(history, donchian_entry, donchian_exit)
    elif backtest_strategy == "buy_hold":
        signals_full = _buy_hold_signals(history)
    else:
        raise BacktestError(f"Unknown strategy: '{backtest_strategy}'.")

    prices = history.loc[str(start_date) :]
    if prices.empty:
        raise BacktestError(
            f"No {commodity_chosen} price data between {start_date} and {end_date}."
        )

    signals = signals_full.reindex(prices.index).astype(int)

    return _build_trades(
        signals=signals,
        prices=prices,
        commodity_chosen=commodity_chosen,
        tons_conversion=tons_conversion,
        contract_size=contract_size,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


# ---------------------------------------------------------------------------
# Signal generators (return an int Series of +1=buy, -1=sell, 0=hold)
# ---------------------------------------------------------------------------


def _tsmom_signals(prices: pd.Series, lookback: int) -> pd.Series:
    """Long when the price is above its level ``lookback`` sessions ago, at month ends.

    Uses the price difference, not the ratio, so a negative price cannot flip the sign.
    """
    up = prices - prices.shift(lookback)
    periods = pd.Series(pd.DatetimeIndex(prices.index).to_period("M"), index=prices.index)
    month_end = periods != periods.shift(-1)
    sig = pd.Series(0, index=prices.index, dtype=int)
    sig[month_end & up.notna() & (up > 0)] = 1
    sig[month_end & up.notna() & (up <= 0)] = -1
    return sig


def _donchian_signals(prices: pd.Series, entry: int, exit_: int) -> pd.Series:
    """Long on a close above the prior ``entry`` high, flat on a close below the prior ``exit_`` low."""
    upper = prices.rolling(entry).max().shift(1)
    lower = prices.rolling(exit_).min().shift(1)
    return (prices > upper).astype(int) - (prices < lower).astype(int)


def _buy_hold_signals(prices: pd.Series) -> pd.Series:
    """Always long: one fill at the first opportunity, never closed."""
    return pd.Series(1, index=prices.index, dtype=int)


# ---------------------------------------------------------------------------
# State machine: convert raw signals to a trade log
# ---------------------------------------------------------------------------


def _build_trades(
    signals: pd.Series,
    prices: pd.Series,
    commodity_chosen: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[pd.DataFrame, bool]:
    trade_type: list[str] = []
    trade_idx: list[int] = []
    in_trade = False
    entry_price = 0.0

    # Iterate over signal bars [0 … N-2]; execution is always at the NEXT bar
    # (T+1 next-close execution — no same-bar look-ahead).
    for i in range(len(signals) - 1):
        sig = int(signals.iloc[i])
        exec_price = float(prices.iloc[i + 1])  # fill at next day's close

        if sig > 0 and not in_trade:
            trade_type.append("buy")
            trade_idx.append(i + 1)
            in_trade = True
            entry_price = exec_price

        elif in_trade:
            forced_exit = False
            if stop_loss_pct is not None and exec_price <= entry_price * (1 - stop_loss_pct / 100):
                forced_exit = True
            if take_profit_pct is not None and exec_price >= entry_price * (
                1 + take_profit_pct / 100
            ):
                forced_exit = True

            if forced_exit or sig < 0:
                trade_type.append("sell")
                trade_idx.append(i + 1)
                in_trade = False

    position_open = in_trade

    if not trade_idx:
        empty = pd.DataFrame(columns=["trade_price", "position", "quantity", "VaR_95"])
        empty.index.name = "date"
        return empty, False

    idx_labels = prices.index[trade_idx]

    df_trades = pd.DataFrame(
        {
            "trade_price": prices.iloc[np.array(trade_idx)].values,
            "position": trade_type,
            "quantity": [1 if t == "buy" else -1 for t in trade_type],
        },
        index=idx_labels,
    )
    df_trades.index.name = "date"

    # Historical VaR 95% — computed per trade using only returns available up
    # to that trade date (no look-ahead bias).
    log_returns = pd.Series(np.log(prices / prices.shift(1))).dropna()
    var_confidence_pct = (1 - config.VAR_CONFIDENCE_LEVEL) * 100  # e.g. 5.0
    var_list: list[float] = []
    for trade_date in idx_labels:
        hist = log_returns.loc[:trade_date]
        var_pct = abs(float(np.percentile(hist, var_confidence_pct))) if len(hist) >= 20 else 0.0
        trade_price = float(prices.loc[trade_date])
        notional = contract_size * trade_price * tons_conversion[commodity_chosen]
        var_list.append(round(var_pct * notional, 2))
    df_trades["VaR_95"] = var_list

    return df_trades, position_open
