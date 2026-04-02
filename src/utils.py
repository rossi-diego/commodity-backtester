"""Trade-level PnL calculation and performance analytics."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "pnl_trades",
    "backtest_performance",
    "backtest_performance_extended",
    "strategy_describe",
]

ANNUALISATION_FACTOR = np.sqrt(config.TRADING_DAYS_PER_YEAR)


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
    commission_per_trade: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Calculate round-trip PnL for every completed buy/sell pair.

    Parameters
    ----------
    commission_per_trade:
        Fixed USD commission charged per trade leg (applied to both buy and sell).
    slippage_pct:
        Slippage as a percentage of trade price (0.05 = 0.05 %).
        Buy price is increased by slippage; sell price is decreased.

    Notes
    -----
    PnL formula (per round trip)::

        pnl = (sell_price_$/MT − buy_price_$/MT) × contract_size_MT
              − 2 × commission_per_trade

    where ``price_$/MT = native_price × tons_conversion``.
    ``contract_size`` is already expressed in metric tons; no further
    conversion is applied to it.
    """
    df_trades = df_trades.copy()
    mtm_trade: dict[str, object] | None = None

    if df_trades.empty:
        return df_trades, mtm_trade

    f = tons_conversion[commodity_chosen]
    slip = slippage_pct / 100.0
    df_trades["pnl_usd"] = 0.0

    for i in range(0, len(df_trades) - 1, 2):
        buy_row = df_trades.iloc[i]
        sell_row = df_trades.iloc[i + 1]

        # Adjust fill prices for slippage
        buy_price_native = float(buy_row["trade_price"]) * (1 + slip)
        sell_price_native = float(sell_row["trade_price"]) * (1 - slip)

        # Convert to $/MT
        buy_mt = buy_price_native * f
        sell_mt = sell_price_native * f

        # PnL = price_diff_$/MT × contract_size_MT − commissions
        # contract_size is in MT — no further conversion factor needed.
        pnl = (sell_mt - buy_mt) * contract_size - 2 * commission_per_trade
        df_trades.at[sell_row.name, "pnl_usd"] = round(pnl, 2)

    df_trades["pnl_usd_cumsum"] = df_trades["pnl_usd"].cumsum()

    # Mark-to-market for unclosed long
    if position_open and not df_trades.empty and df_trades.iloc[-1]["position"] == "buy":
        last_buy_native = float(df_trades.iloc[-1]["trade_price"]) * (1 + slip)
        last_price_native = float(df_prices[commodity_chosen].iloc[-1])
        pnl_mtm = (last_price_native - last_buy_native) * f * contract_size - commission_per_trade
        mtm_trade = {"date": df_prices.index[-1], "pnl_usd": round(pnl_mtm, 2)}

    return df_trades, mtm_trade


# ---------------------------------------------------------------------------
# Performance metrics — standard
# ---------------------------------------------------------------------------


def backtest_performance(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None = None,
    contract_size: float | None = None,
    tons_conversion: dict[str, float] | None = None,
    commodity_chosen: str | None = None,
    position_open: bool = False,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """Core performance summary (12 metrics)."""
    return _build_metrics(
        df_trades,
        df_prices,
        mtm_trade,
        contract_size,
        tons_conversion,
        commodity_chosen,
        position_open,
        extended=False,
        initial_capital=initial_capital,
    )


def backtest_performance_extended(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None = None,
    contract_size: float | None = None,
    tons_conversion: dict[str, float] | None = None,
    commodity_chosen: str | None = None,
    position_open: bool = False,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """Extended performance summary with Sortino, Calmar, Profit Factor, avg win/loss."""
    return _build_metrics(
        df_trades,
        df_prices,
        mtm_trade,
        contract_size,
        tons_conversion,
        commodity_chosen,
        position_open,
        extended=True,
        initial_capital=initial_capital,
    )


# ---------------------------------------------------------------------------
# Descriptive statistics across all spreads
# ---------------------------------------------------------------------------


def strategy_describe(
    df: pd.DataFrame,
    tons_conversion: dict[str, float],
    backtest_strategy: str | None = None,
) -> pd.DataFrame:
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

    return pd.DataFrame(summary).round(4) if summary else pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


def _build_metrics(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None,
    contract_size: float | None,
    tons_conversion: dict[str, float] | None,
    commodity_chosen: str | None,
    position_open: bool,
    extended: bool,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    tons_conv = tons_conversion or {}
    com = commodity_chosen or ""
    c_size = contract_size or 0.0

    # ------------------------------------------------------------------
    # Initial capital — one contract notional at the first price in the
    # backtest window.  Used to normalise returns so that Sharpe, Sortino,
    # and Calmar are dimensionally consistent.
    # ------------------------------------------------------------------
    if initial_capital is None and com and tons_conv and c_size and not df_prices.empty:
        first_price = float(df_prices[com].dropna().iloc[0])
        initial_capital = c_size * tons_conv[com] * first_price
    # Fallback: if we still can't compute it, use 1 (prevents /0 errors but
    # the caller should always supply the relevant parameters).
    if not initial_capital:
        initial_capital = 1.0

    total_complete = int(len(df_trades) / 2)
    realized_pnl = float(df_trades["pnl_usd"].sum())
    mtm_pnl = float(str(mtm_trade["pnl_usd"])) if mtm_trade else 0.0
    total_pnl = realized_pnl + mtm_pnl

    wins = int((df_trades["pnl_usd"] > 0).sum())
    losses = int((df_trades["pnl_usd"] < 0).sum())
    win_rate = wins / total_complete if total_complete > 0 else 0.0

    # ------------------------------------------------------------------
    # Max drawdown — in USD (for display) and as % of initial capital
    # (for Calmar).
    # ------------------------------------------------------------------
    cum = df_trades["pnl_usd_cumsum"]
    cum_arr = np.asarray(cum, dtype=float)
    max_drawdown = float((np.maximum.accumulate(cum_arr) - cum_arr).max()) if len(cum) else 0.0
    max_drawdown_pct = max_drawdown / initial_capital if initial_capital > 0 else 0.0

    # ------------------------------------------------------------------
    # Daily returns — normalised by initial capital so they are unitless
    # fractions suitable for Sharpe / Sortino.
    # ------------------------------------------------------------------
    daily_equity = _build_daily_equity(df_trades, df_prices, mtm_trade)
    daily_pnl = daily_equity.diff().fillna(0.0)
    daily_returns = (daily_pnl / initial_capital).replace([np.inf, -np.inf], np.nan).dropna()

    sharpe = _sharpe(daily_returns)

    gross_profit = float(df_trades.loc[df_trades["pnl_usd"] > 0, "pnl_usd"].sum())
    gross_loss = abs(float(df_trades.loc[df_trades["pnl_usd"] < 0, "pnl_usd"].sum()))

    gross_exposure = 0.0
    if (
        position_open
        and com
        and tons_conv
        and not df_trades.empty
        and df_trades.iloc[-1]["position"] == "buy"
    ):
        gross_exposure = c_size * tons_conv[com] * float(df_prices[com].iloc[-1])

    var_95 = 0.0
    if gross_exposure and com:
        log_ret = pd.Series(np.log(df_prices[com] / df_prices[com].shift(1))).dropna()
        var_pct = (1 - config.VAR_CONFIDENCE_LEVEL) * 100
        var_95 = abs(float(np.percentile(log_ret, var_pct))) * gross_exposure

    rows: list[tuple[str, object]] = [
        ("Total Buys", int((df_trades["position"] == "buy").sum())),
        ("Total Sells", int((df_trades["position"] == "sell").sum())),
        ("Complete Trades", total_complete),
        (
            "Open Positions",
            int((df_trades["position"] == "buy").sum())
            - int((df_trades["position"] == "sell").sum()),
        ),
        ("Realized Profit (USD)", realized_pnl),
        ("MTM Adjustment (USD)", mtm_pnl),
        ("Total Profit (USD)", total_pnl),
        ("Win Rate (%)", win_rate * 100),
        ("Max Drawdown (USD)", max_drawdown),
        ("Sharpe Ratio", sharpe),
        ("Best Trade (USD)", float(df_trades["pnl_usd"].max())),
        ("Worst Trade (USD)", float(df_trades["pnl_usd"].min())),
        (
            "Backtest Duration (days)",
            int((df_prices.index[-1] - df_prices.index[0]).days) if len(df_prices) > 1 else 0,
        ),
        ("Gross Exposure (USD)", gross_exposure),
        ("VaR 95% — Historical (USD)", var_95),
    ]

    if extended:
        annual_return = _annualised_return(daily_equity, initial_capital)
        sortino = _sortino(daily_returns)
        calmar = _calmar(annual_return, max_drawdown_pct)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        winning_trades = df_trades[df_trades["pnl_usd"] > 0]["pnl_usd"]
        losing_trades = df_trades[df_trades["pnl_usd"] < 0]["pnl_usd"]
        avg_win = float(winning_trades.mean()) if not winning_trades.empty else 0.0
        avg_loss = float(losing_trades.mean()) if not losing_trades.empty else 0.0
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else float("inf")

        rows += [
            ("Annualised Return (%)", round(annual_return * 100, 4)),
            ("Sortino Ratio", sortino),
            ("Calmar Ratio", calmar),
            ("Profit Factor", profit_factor),
            ("Recovery Factor", recovery_factor),
            ("Winning Trades", wins),
            ("Losing Trades", losses),
            ("Average Win (USD)", avg_win),
            ("Average Loss (USD)", avg_loss),
        ]

    df_out = pd.DataFrame(rows, columns=["Metric", "Value"])
    df_out["Value"] = df_out["Value"].apply(lambda v: round(v, 4) if isinstance(v, float) else v)
    return df_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_daily_equity(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None,
) -> pd.Series:
    if df_trades.empty or "pnl_usd_cumsum" not in df_trades.columns:
        return pd.Series(dtype=float)
    equity = df_trades["pnl_usd_cumsum"].reindex(df_prices.index).ffill().fillna(0.0)
    if mtm_trade:
        mtm_ts = pd.Timestamp(str(mtm_trade["date"]))
        if mtm_ts in equity.index:
            equity.loc[mtm_ts:] += float(str(mtm_trade["pnl_usd"]))
    return equity


def _annualised_return(equity: pd.Series, initial_capital: float) -> float:
    """CAGR expressed as a fraction (0.12 = 12 %).

    Uses initial_capital as the denominator so the equity curve's zero-start
    does not break the calculation.
    """
    if len(equity) < 2 or initial_capital <= 0:
        return 0.0
    total_return = equity.iloc[-1] / initial_capital  # total PnL / initial notional
    years = len(equity) / config.TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def _sharpe(
    daily_returns: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
) -> float:
    """Annualised Sharpe ratio.

    daily_returns must be dimensionless (pnl / initial_capital).
    """
    std = float(daily_returns.std())
    if std == 0 or daily_returns.empty:
        return float("nan")
    return float(
        (daily_returns.mean() - risk_free / config.TRADING_DAYS_PER_YEAR)
        / std
        * ANNUALISATION_FACTOR
    )


def _sortino(
    daily_returns: pd.Series,
    risk_free: float = config.RISK_FREE_RATE,
) -> float:
    """Annualised Sortino ratio (downside deviation of negative returns only).

    daily_returns must be dimensionless (pnl / initial_capital).

    Returns ``inf`` when there are no negative return days (no downside risk),
    which is the mathematically correct limit.  Returns ``nan`` only when the
    input series is empty (undefined).
    """
    if daily_returns.empty:
        return float("nan")
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return float("inf")  # no losing days → perfect Sortino
    downside_std = float(downside.std())
    if downside_std == 0 or np.isnan(downside_std):
        return float("inf")
    excess = daily_returns.mean() - risk_free / config.TRADING_DAYS_PER_YEAR
    return float(excess / downside_std * ANNUALISATION_FACTOR)


def _calmar(annualised_return: float, max_drawdown_pct: float) -> float:
    """Calmar ratio = annualised return / max drawdown (both dimensionless fractions)."""
    if max_drawdown_pct <= 0:
        return float("inf")
    return round(annualised_return / max_drawdown_pct, 4)
