"""
Trade-level PnL calculation and performance analytics.

All metrics follow standard quantitative finance conventions:

- **Sharpe ratio** — annualised, computed on daily equity-curve returns
  (not per-trade PnL, which is the incorrect but common approach)
- **Sortino ratio** — same as Sharpe but uses downside deviation
- **Calmar ratio** — annualised return / maximum drawdown
- **Profit factor** — gross profit / |gross loss|
- **Recovery factor** — total profit / maximum drawdown
"""

from __future__ import annotations

import datetime
from itertools import permutations

import numpy as np
import pandas as pd

from .constants import commodities_dict, contract_sizes, tons_conversion

ANNUALISATION_FACTOR = np.sqrt(252)


# ---------------------------------------------------------------------------
# PnL engine
# ---------------------------------------------------------------------------

def pnl_trades(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    commodity_chosen: str,
    tons_conversion: dict[str, float],
    contract_size: float,
    position_open: bool,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Calculate round-trip PnL for every completed buy/sell pair.

    Parameters
    ----------
    df_trades:
        Output of :func:`strategy.backtest`.
    df_prices:
        Full price DataFrame (used for MTM valuation).
    commodity_chosen:
        Primary commodity display name.
    tons_conversion:
        Price-to-USD-per-MT factors.
    contract_size:
        Metric tons per contract.
    position_open:
        Whether the last trade is an unclosed long.

    Returns
    -------
    df_trades:
        Input frame augmented with ``pnl_usd`` and ``pnl_usd_cumsum`` columns.
    mtm_trade:
        ``{"date": ..., "pnl_usd": float}`` for the open position, or ``None``.
    """
    df_trades = df_trades.copy()
    mtm_trade: dict[str, object] | None = None

    if df_trades.empty:
        return df_trades, mtm_trade

    contract_tons = contract_size * tons_conversion[commodity_chosen]
    df_trades["pnl_usd"] = 0.0

    # Pair up buys and sells (indices 0,1 | 2,3 | ...)
    for i in range(0, len(df_trades) - 1, 2):
        buy_row  = df_trades.iloc[i]
        sell_row = df_trades.iloc[i + 1]
        buy_price_mt  = float(buy_row["trade_price"])  * tons_conversion[commodity_chosen]
        sell_price_mt = float(sell_row["trade_price"]) * tons_conversion[commodity_chosen]
        pnl = (sell_price_mt - buy_price_mt) * contract_tons
        df_trades.at[sell_row.name, "pnl_usd"] = round(pnl, 2)

    df_trades["pnl_usd_cumsum"] = df_trades["pnl_usd"].cumsum()

    # ── Mark-to-market for unclosed long ─────────────────────────────────────
    if position_open and df_trades.iloc[-1]["position"] == "buy":
        last_buy_price = float(df_trades.iloc[-1]["trade_price"])
        last_price     = float(df_prices[commodity_chosen].iloc[-1])
        buy_mt   = last_buy_price * tons_conversion[commodity_chosen]
        last_mt  = last_price     * tons_conversion[commodity_chosen]
        mtm_pnl  = (last_mt - buy_mt) * contract_tons
        mtm_trade = {
            "date":    df_prices.index[-1],
            "pnl_usd": round(mtm_pnl, 2),
        }

    return df_trades, mtm_trade


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def backtest_performance(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None = None,
    contract_size: float | None = None,
    tons_conversion: dict[str, float] | None = None,
    commodity_chosen: str | None = None,
    position_open: bool = False,
) -> pd.DataFrame:
    """Compute a comprehensive performance summary table.

    Metrics
    -------
    Trade counts, realized/MTM/total profit, win rate, max drawdown,
    Sharpe (annualised, daily returns), Sortino, Calmar, Profit Factor,
    Recovery Factor, best/worst trade, mean duration, gross exposure, VaR 95%.

    Returns
    -------
    DataFrame with columns ``["Metric", "Value"]``.
    """
    tons_conv  = tons_conversion or {}
    com        = commodity_chosen or ""
    c_size     = contract_size or 0.0

    total_complete = int(len(df_trades) / 2)
    realized_pnl   = float(df_trades["pnl_usd"].sum())
    mtm_pnl        = float(mtm_trade["pnl_usd"]) if mtm_trade else 0.0
    total_pnl      = realized_pnl + mtm_pnl

    wins   = int((df_trades["pnl_usd"] > 0).sum())
    losses = int((df_trades["pnl_usd"] < 0).sum())
    win_rate = wins / total_complete if total_complete > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cum = df_trades["pnl_usd_cumsum"]
    running_max  = np.maximum.accumulate(cum.values)
    drawdown_ser = running_max - cum.values
    max_drawdown = float(drawdown_ser.max()) if len(drawdown_ser) else 0.0

    # ── Daily equity curve (needed for Sharpe / Sortino / Calmar) ─────────────
    daily_equity  = _build_daily_equity(df_trades, df_prices, mtm_trade)
    daily_returns = daily_equity.pct_change().dropna()
    annual_return = _annualised_return(daily_equity)

    sharpe  = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)
    calmar  = _calmar(annual_return, max_drawdown)

    # ── Profit factor & recovery factor ──────────────────────────────────────
    gross_profit = float(df_trades.loc[df_trades["pnl_usd"] > 0, "pnl_usd"].sum())
    gross_loss   = abs(float(df_trades.loc[df_trades["pnl_usd"] < 0, "pnl_usd"].sum()))
    profit_factor   = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    recovery_factor = total_pnl / max_drawdown  if max_drawdown > 0 else float("inf")

    # ── Trade-level stats ─────────────────────────────────────────────────────
    best_trade  = float(df_trades["pnl_usd"].max())
    worst_trade = float(df_trades["pnl_usd"].min())
    mean_duration_days = _mean_duration(df_trades)

    backtest_days = int(
        (df_prices.index[-1] - df_prices.index[0]).days
    ) if len(df_prices) > 1 else 0

    # ── Gross exposure & VaR ─────────────────────────────────────────────────
    gross_exposure = 0.0
    if position_open and com and tons_conv and not df_trades.empty:
        if df_trades.iloc[-1]["position"] == "buy":
            last_price     = float(df_prices[com].iloc[-1])
            gross_exposure = c_size * tons_conv[com] * last_price

    var_95 = 0.0
    if gross_exposure and com:
        log_ret = np.log(
            df_prices[com] / df_prices[com].shift(1)
        ).dropna()
        var_95 = abs(float(np.percentile(log_ret, 5))) * gross_exposure

    rows: list[tuple[str, object]] = [
        ("Total Buys",                    int((df_trades["position"] == "buy").sum())),
        ("Total Sells",                   int((df_trades["position"] == "sell").sum())),
        ("Complete Trades",               total_complete),
        ("Open Positions",                int((df_trades["position"] == "buy").sum()) - int((df_trades["position"] == "sell").sum())),
        ("Realized Profit (USD)",         realized_pnl),
        ("MTM Adjustment (USD)",          mtm_pnl),
        ("Total Profit (USD)",            total_pnl),
        ("Win Rate (%)",                  win_rate * 100),
        ("Winning Trades",                wins),
        ("Losing Trades",                 losses),
        ("Max Drawdown (USD)",            max_drawdown),
        ("Sharpe Ratio (ann.)",           sharpe),
        ("Sortino Ratio (ann.)",          sortino),
        ("Calmar Ratio",                  calmar),
        ("Profit Factor",                 profit_factor),
        ("Recovery Factor",               recovery_factor),
        ("Best Trade (USD)",              best_trade),
        ("Worst Trade (USD)",             worst_trade),
        ("Mean Trade Duration (days)",    mean_duration_days),
        ("Backtest Duration (days)",      backtest_days),
        ("Gross Exposure (USD)",          gross_exposure),
        ("VaR 95% — Historical (USD)",   var_95),
    ]

    df_out = pd.DataFrame(rows, columns=["Metric", "Value"])
    # Round only numeric rows
    df_out["Value"] = df_out["Value"].apply(
        lambda v: round(v, 4) if isinstance(v, float) else v
    )
    return df_out


# ---------------------------------------------------------------------------
# Descriptive statistics for all spread ratios
# ---------------------------------------------------------------------------

def strategy_describe(
    df: pd.DataFrame,
    tons_conversion: dict[str, float],
    backtest_strategy: str | None = None,
) -> pd.DataFrame:
    """Return summary statistics for every ordered pair of commodities.

    Only applicable for the ``"ratio"`` strategy.
    """
    if backtest_strategy != "ratio":
        return pd.DataFrame(index=df.index)

    summary: list[pd.Series] = []
    for col_a, col_b in permutations(df.columns, 2):
        if col_a in tons_conversion and col_b in tons_conversion:
            ratio = (df[col_a] * tons_conversion[col_a]) / (df[col_b] * tons_conversion[col_b])
            stats = ratio.describe()
            stats["coefficient_variation"] = stats["std"] / stats["mean"]
            stats.name = f"{col_a}/{col_b}"
            summary.append(stats)

    if not summary:
        return pd.DataFrame()

    return pd.DataFrame(summary).round(4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_daily_equity(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None,
) -> pd.Series:
    """Reindex trade PnL to a daily equity curve over the full price history."""
    if df_trades.empty or "pnl_usd_cumsum" not in df_trades.columns:
        return pd.Series(dtype=float)

    equity = df_trades["pnl_usd_cumsum"].reindex(df_prices.index, method="ffill").fillna(0.0)

    if mtm_trade:
        mtm_date = pd.Timestamp(str(mtm_trade["date"]))
        if mtm_date in equity.index:
            equity.loc[mtm_date:] += float(str(mtm_trade["pnl_usd"]))

    return equity


def _annualised_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    total_return = (equity.iloc[-1] - equity.iloc[0]) / abs(equity.iloc[0])
    years = len(equity) / 252
    return float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0


def _sharpe(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    std = float(daily_returns.std())
    if std == 0 or daily_returns.empty:
        return float("nan")
    excess = daily_returns - risk_free / 252
    return float(excess.mean() / std * ANNUALISATION_FACTOR)


def _sortino(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std())
    if downside_std == 0 or daily_returns.empty:
        return float("nan")
    excess = daily_returns.mean() - risk_free / 252
    return float(excess / downside_std * ANNUALISATION_FACTOR)


def _calmar(annualised_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return float("inf")
    return round(annualised_return / (max_drawdown / 1e6), 4)  # normalise


def _mean_duration(df_trades: pd.DataFrame) -> float:
    """Return mean holding period in days between consecutive buy→sell pairs."""
    if df_trades.empty or len(df_trades) < 2:
        return float("nan")
    diffs = df_trades.index.to_series().diff().dropna()
    mean_td = diffs.mean()
    return float(mean_td.days) if hasattr(mean_td, "days") else float("nan")
