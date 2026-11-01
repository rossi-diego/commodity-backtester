"""Runs one backtest for the app: the window slice, the engine, P&L and summary.

All window logic lives here so that it can be tested without the UI. Marking an
open position on the window, and not on the full history, depends on it.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Final

import pandas as pd

from .constants import contract_sizes, tons_conversion
from .strategy import backtest
from .utils import BacktestSummary, backtest_performance, pnl_trades, summarize_backtest

__all__ = ["BacktestResult", "MIN_ROUND_TRIPS_FOR_SHARPE", "run_backtest"]

# Below this, Sharpe is dominated by a few realised jumps. For one open position it
# reduces mechanically to +/-sqrt(252/n), whatever the result was.
MIN_ROUND_TRIPS_FOR_SHARPE: Final[int] = 10


@dataclass(frozen=True)
class BacktestResult:
    """Everything the UI shows for one market, rule and window."""

    window: pd.Series
    trades: pd.DataFrame
    mtm: dict[str, object] | None
    summary: BacktestSummary
    sharpe: float | None


def run_backtest(
    rule: str,
    market: str,
    start: datetime.date | str,
    end: datetime.date | str,
    prices: pd.DataFrame,
    commission: float,
    slippage: float,
) -> BacktestResult:
    """Run ``rule`` on ``market`` over [start, end], one contract, with the given costs.

    Raises
    ------
    BacktestError
        If the market has no prices in the window.
    """
    raw_trades, position_open = backtest(
        backtest_strategy=rule,
        start_date=start,
        end_date=end,
        df=prices,
        commodity_chosen=market,
        tons_conversion=tons_conversion,
        contract_size=contract_sizes[market],
    )

    window = prices[market].loc[str(start) : str(end)].dropna()
    window_frame = window.to_frame(market)

    trades, mtm = pnl_trades(
        df_trades=raw_trades,
        df_prices=window_frame,
        commodity_chosen=market,
        tons_conversion=tons_conversion,
        contract_size=contract_sizes[market],
        position_open=position_open,
        commission_per_trade=commission,
        slippage_pct=slippage,
    )
    summary = summarize_backtest(trades, window, mtm, position_open)

    sharpe = None
    if rule != "buy_hold" and summary.round_trips >= MIN_ROUND_TRIPS_FOR_SHARPE:
        sharpe = _indicative_sharpe(trades, window_frame, mtm, market, position_open)

    return BacktestResult(window=window, trades=trades, mtm=mtm, summary=summary, sharpe=sharpe)


def _indicative_sharpe(
    trades: pd.DataFrame,
    window_frame: pd.DataFrame,
    mtm: dict[str, object] | None,
    market: str,
    position_open: bool,
) -> float | None:
    """Return the engine's Sharpe on capital sized at the first positive price of the window."""
    positive = window_frame[market][window_frame[market] > 0]
    if positive.empty:
        return None
    metrics = backtest_performance(
        df_trades=trades,
        df_prices=window_frame,
        mtm_trade=mtm,
        contract_size=contract_sizes[market],
        tons_conversion=tons_conversion,
        commodity_chosen=market,
        position_open=position_open,
        initial_capital=contract_sizes[market] * tons_conversion[market] * float(positive.iloc[0]),
    )
    sharpe = float(metrics.loc[metrics["Metric"] == "Sharpe Ratio", "Value"].iloc[0])
    return sharpe if math.isfinite(sharpe) else None
