"""Trade-level PnL calculation and performance analytics."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

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
    commission_per_trade: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Calculate round-trip PnL for every completed buy/sell pair.

    Parameters
    ----------
    commission_per_trade:
        Fixed USD commission charged per trade leg (applied to both buy and sell).
    slippage_pct:
        Slippage as a percentage of trade price (0.05 = 0.05%).
        Buy price is increased by slippage; sell price is decreased.
    """
    df_trades = df_trades.copy()
    mtm_trade: dict[str, object] | None = None

    if df_trades.empty:
        return df_trades, mtm_trade

    contract_tons = contract_size * tons_conversion[commodity_chosen]
    slip = slippage_pct / 100.0
    df_trades["pnl_usd"] = 0.0

    for i in range(0, len(df_trades) - 1, 2):
        buy_row = df_trades.iloc[i]
        sell_row = df_trades.iloc[i + 1]

        buy_price = float(buy_row["trade_price"]) * (1 + slip)
        sell_price = float(sell_row["trade_price"]) * (1 - slip)

        buy_mt = buy_price * tons_conversion[commodity_chosen]
        sell_mt = sell_price * tons_conversion[commodity_chosen]

        pnl = (sell_mt - buy_mt) * contract_tons - 2 * commission_per_trade
        df_trades.at[sell_row.name, "pnl_usd"] = round(pnl, 2)

    df_trades["pnl_usd_cumsum"] = df_trades["pnl_usd"].cumsum()

    # Mark-to-market for unclosed long
    if position_open and not df_trades.empty and df_trades.iloc[-1]["position"] == "buy":
        last_buy = float(df_trades.iloc[-1]["trade_price"]) * (1 + slip)
        last_price = float(df_prices[commodity_chosen].iloc[-1])
        pnl_mtm = (last_price - last_buy) * tons_conversion[
            commodity_chosen
        ] * contract_tons - commission_per_trade
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
    )


def backtest_performance_extended(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: dict[str, object] | None = None,
    contract_size: float | None = None,
    tons_conversion: dict[str, float] | None = None,
    commodity_chosen: str | None = None,
    position_open: bool = False,
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
) -> pd.DataFrame:
    tons_conv = tons_conversion or {}
    com = commodity_chosen or ""
    c_size = contract_size or 0.0

    total_complete = int(len(df_trades) / 2)
    realized_pnl = float(df_trades["pnl_usd"].sum())
    mtm_pnl = float(str(mtm_trade["pnl_usd"])) if mtm_trade else 0.0
    total_pnl = realized_pnl + mtm_pnl

    wins = int((df_trades["pnl_usd"] > 0).sum())
    losses = int((df_trades["pnl_usd"] < 0).sum())
    win_rate = wins / total_complete if total_complete > 0 else 0.0

    cum = df_trades["pnl_usd_cumsum"]
    cum_arr = np.asarray(cum, dtype=float)
    max_drawdown = float((np.maximum.accumulate(cum_arr) - cum_arr).max()) if len(cum) else 0.0

    daily_equity = _build_daily_equity(df_trades, df_prices, mtm_trade)
    daily_returns = daily_equity.pct_change().dropna()
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
        var_95 = abs(float(np.percentile(log_ret, 5))) * gross_exposure

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
        annual_return = _annualised_return(daily_equity)
        sortino = _sortino(daily_returns)
        calmar = _calmar(annual_return, max_drawdown)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        winning_trades = df_trades[df_trades["pnl_usd"] > 0]["pnl_usd"]
        losing_trades = df_trades[df_trades["pnl_usd"] < 0]["pnl_usd"]
        avg_win = float(winning_trades.mean()) if not winning_trades.empty else 0.0
        avg_loss = float(losing_trades.mean()) if not losing_trades.empty else 0.0

        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else float("inf")

        rows += [
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


def _annualised_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    total = (equity.iloc[-1] - equity.iloc[0]) / abs(equity.iloc[0])
    years = len(equity) / 252
    return float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0


def _sharpe(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    std = float(daily_returns.std())
    if std == 0 or daily_returns.empty:
        return float("nan")
    return float((daily_returns.mean() - risk_free / 252) / std * ANNUALISATION_FACTOR)


def _sortino(daily_returns: pd.Series) -> float:
    downside_std = float(daily_returns[daily_returns < 0].std())
    if downside_std == 0 or daily_returns.empty:
        return float("nan")
    return float(daily_returns.mean() / downside_std * ANNUALISATION_FACTOR)


def _calmar(annualised_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return float("inf")
    return round(annualised_return / max_drawdown, 4)
