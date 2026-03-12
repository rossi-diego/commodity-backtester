"""
Advanced Performance Metrics Module.

This module provides comprehensive trading performance analytics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- Rolling performance
- Monte Carlo simulation
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a trading strategy.

    Contains all key performance indicators used by professional traders
    and portfolio managers.
    """
    # Basic Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L Metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    gross_profit: float
    gross_loss: float

    # Average Trade Metrics
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winning_trade: float
    largest_losing_trade: float

    # Risk Metrics
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # days
    avg_drawdown: float

    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Ratios
    profit_factor: float
    payoff_ratio: float  # avg win / avg loss
    expectancy: float  # expected $ per trade

    # Exposure
    total_exposure_time: float  # % of time in market
    avg_holding_period: float  # days
    max_holding_period: float
    min_holding_period: float

    # Statistical
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Streak Analysis
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Annualized
    annualized_return: float
    annualized_volatility: float


def calculate_comprehensive_metrics(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    commodity_chosen: str,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.05
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for a backtest.

    Args:
        df_trades: DataFrame with trade records including 'pnl_usd' column
        df_prices: DataFrame with price history
        commodity_chosen: Name of the traded commodity
        initial_capital: Starting capital for return calculations
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PerformanceMetrics dataclass with all calculated metrics
    """
    if df_trades.empty or "pnl_usd" not in df_trades.columns:
        return _empty_metrics()

    # Separate buy and sell trades
    sells = df_trades[df_trades["position"] == "sell"]

    # P&L series (from sell trades only, where P&L is recorded)
    pnl_series = sells["pnl_usd"].dropna()

    # Basic counts
    total_trades = len(sells)  # Complete round trips
    winning_trades = (pnl_series > 0).sum()
    losing_trades = (pnl_series < 0).sum()

    if total_trades == 0:
        return _empty_metrics()

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # P&L calculations
    total_pnl = pnl_series.sum()
    realized_pnl = total_pnl
    unrealized_pnl = 0.0  # Will be added from MTM if available

    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())

    # Average trade metrics
    avg_trade_pnl = pnl_series.mean() if len(pnl_series) > 0 else 0.0

    winning_pnl = pnl_series[pnl_series > 0]
    losing_pnl = pnl_series[pnl_series < 0]

    avg_winning_trade = winning_pnl.mean() if len(winning_pnl) > 0 else 0.0
    avg_losing_trade = abs(losing_pnl.mean()) if len(losing_pnl) > 0 else 0.0

    largest_winning_trade = pnl_series.max() if len(pnl_series) > 0 else 0.0
    largest_losing_trade = abs(pnl_series.min()) if len(pnl_series) > 0 else 0.0

    # Drawdown analysis
    cum_pnl = df_trades["pnl_usd_cumsum"] if "pnl_usd_cumsum" in df_trades.columns else pnl_series.cumsum()
    drawdown_stats = _calculate_drawdown_stats(cum_pnl, initial_capital)

    # Risk-adjusted returns
    daily_returns = _calculate_daily_returns(df_trades, df_prices, commodity_chosen, initial_capital)
    risk_metrics = _calculate_risk_adjusted_returns(daily_returns, risk_free_rate)

    # Ratios
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    payoff_ratio = avg_winning_trade / avg_losing_trade if avg_losing_trade > 0 else np.inf
    expectancy = (win_rate * avg_winning_trade) - ((1 - win_rate) * avg_losing_trade)

    # Holding period analysis
    holding_stats = _calculate_holding_periods(df_trades)

    # Exposure time
    backtest_days = (df_prices.index[-1] - df_prices.index[0]).days
    days_in_market = _calculate_days_in_market(df_trades)
    exposure_time = days_in_market / backtest_days if backtest_days > 0 else 0.0

    # Statistical measures
    skewness = stats.skew(pnl_series) if len(pnl_series) > 2 else 0.0
    kurtosis = stats.kurtosis(pnl_series) if len(pnl_series) > 3 else 0.0

    # VaR and CVaR
    var_95, cvar_95 = _calculate_var_cvar(pnl_series)

    # Streak analysis
    max_wins, max_losses = _calculate_streaks(pnl_series)

    # Annualized metrics
    total_return = total_pnl / initial_capital
    years = backtest_days / 365.25
    annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
    annualized_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(daily_returns) > 0 else 0.0

    return PerformanceMetrics(
        # Basic Statistics
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,

        # P&L Metrics
        total_pnl=total_pnl,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,

        # Average Trade Metrics
        avg_trade_pnl=avg_trade_pnl,
        avg_winning_trade=avg_winning_trade,
        avg_losing_trade=avg_losing_trade,
        largest_winning_trade=largest_winning_trade,
        largest_losing_trade=largest_losing_trade,

        # Risk Metrics
        max_drawdown=drawdown_stats["max_drawdown"],
        max_drawdown_pct=drawdown_stats["max_drawdown_pct"],
        max_drawdown_duration=drawdown_stats["max_drawdown_duration"],
        avg_drawdown=drawdown_stats["avg_drawdown"],

        # Risk-Adjusted Returns
        sharpe_ratio=risk_metrics["sharpe"],
        sortino_ratio=risk_metrics["sortino"],
        calmar_ratio=(annualized_return / drawdown_stats["max_drawdown"] if drawdown_stats["max_drawdown"] > 0 else 0.0),

        # Ratios
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,

        # Exposure
        total_exposure_time=exposure_time,
        avg_holding_period=holding_stats["avg"],
        max_holding_period=holding_stats["max"],
        min_holding_period=holding_stats["min"],

        # Statistical
        skewness=skewness,
        kurtosis=kurtosis,
        var_95=var_95,
        cvar_95=cvar_95,

        # Streak Analysis
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,

        # Annualized
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
    )


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics object."""
    return PerformanceMetrics(
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
        total_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0,
        gross_profit=0.0, gross_loss=0.0,
        avg_trade_pnl=0.0, avg_winning_trade=0.0, avg_losing_trade=0.0,
        largest_winning_trade=0.0, largest_losing_trade=0.0,
        max_drawdown=0.0, max_drawdown_pct=0.0, max_drawdown_duration=0, avg_drawdown=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
        profit_factor=0.0, payoff_ratio=0.0, expectancy=0.0,
        total_exposure_time=0.0, avg_holding_period=0.0, max_holding_period=0.0, min_holding_period=0.0,
        skewness=0.0, kurtosis=0.0, var_95=0.0, cvar_95=0.0,
        max_consecutive_wins=0, max_consecutive_losses=0,
        annualized_return=0.0, annualized_volatility=0.0,
    )


def _calculate_drawdown_stats(cum_pnl: pd.Series, initial_capital: float) -> dict:
    """Calculate drawdown statistics."""
    if cum_pnl.empty:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_duration": 0, "avg_drawdown": 0.0}

    # Portfolio value
    portfolio_value = initial_capital + cum_pnl

    # Running maximum
    running_max = portfolio_value.cummax()

    # Drawdown in dollars
    drawdown = running_max - portfolio_value
    max_drawdown = drawdown.max()

    # Drawdown in percentage
    drawdown_pct = drawdown / running_max
    max_drawdown_pct = drawdown_pct.max()

    # Average drawdown
    avg_drawdown = drawdown[drawdown > 0].mean() if (drawdown > 0).any() else 0.0

    # Max drawdown duration
    in_drawdown = drawdown > 0
    if in_drawdown.any():
        # Find consecutive drawdown periods
        drawdown_starts = (~in_drawdown).cumsum()
        drawdown_groups = in_drawdown.groupby(drawdown_starts)
        max_duration = max(group.sum() for _, group in drawdown_groups)
    else:
        max_duration = 0

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_duration": max_duration,
        "avg_drawdown": avg_drawdown,
    }


def _calculate_daily_returns(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    commodity_chosen: str,
    initial_capital: float
) -> pd.Series:
    """Calculate daily portfolio returns."""
    if df_trades.empty or "pnl_usd_cumsum" not in df_trades.columns:
        return pd.Series(dtype=float)

    # Create daily portfolio value series
    portfolio_value = initial_capital + df_trades["pnl_usd_cumsum"]

    # Calculate returns
    returns = portfolio_value.pct_change().dropna()

    return returns


def _calculate_risk_adjusted_returns(daily_returns: pd.Series, risk_free_rate: float) -> dict:
    """Calculate risk-adjusted return metrics."""
    if daily_returns.empty or len(daily_returns) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = daily_returns - daily_rf

    # Sharpe Ratio
    if daily_returns.std() != 0:
        sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    # Sortino Ratio (uses downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sortino = 0.0

    # Calmar Ratio (annualized return / max drawdown)
    # Note: requires drawdown from portfolio level
    calmar = 0.0  # Will be calculated separately

    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar}


def _calculate_holding_periods(df_trades: pd.DataFrame) -> dict:
    """Calculate holding period statistics."""
    if df_trades.empty or len(df_trades) < 2:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}

    # Pair up buy/sell trades
    buys = df_trades[df_trades["position"] == "buy"]
    sells = df_trades[df_trades["position"] == "sell"]

    holding_periods = []

    for i in range(min(len(buys), len(sells))):
        buy_date = buys.index[i]
        sell_date = sells.index[i]
        holding_days = (sell_date - buy_date).days
        holding_periods.append(holding_days)

    if not holding_periods:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}

    return {
        "avg": np.mean(holding_periods),
        "max": max(holding_periods),
        "min": min(holding_periods),
    }


def _calculate_days_in_market(df_trades: pd.DataFrame) -> int:
    """Calculate total days with open position."""
    if df_trades.empty:
        return 0

    buys = df_trades[df_trades["position"] == "buy"]
    sells = df_trades[df_trades["position"] == "sell"]

    total_days = 0
    for i in range(min(len(buys), len(sells))):
        buy_date = buys.index[i]
        sell_date = sells.index[i]
        total_days += (sell_date - buy_date).days

    return total_days


def _calculate_var_cvar(pnl_series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Value at Risk and Conditional VaR."""
    if pnl_series.empty:
        return 0.0, 0.0

    alpha = 1 - confidence
    var = np.percentile(pnl_series, alpha * 100)
    cvar = pnl_series[pnl_series <= var].mean()

    return abs(var), abs(cvar) if not np.isnan(cvar) else abs(var)


def _calculate_streaks(pnl_series: pd.Series) -> tuple[int, int]:
    """Calculate maximum winning and losing streaks."""
    if pnl_series.empty:
        return 0, 0

    wins = (pnl_series > 0).astype(int)
    losses = (pnl_series < 0).astype(int)

    # Calculate streaks
    def max_streak(series: pd.Series) -> int:
        if series.empty:
            return 0
        max_count = 0
        current_count = 0
        for val in series:
            if val == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    return max_streak(wins), max_streak(losses)


def metrics_to_dataframe(metrics: PerformanceMetrics) -> pd.DataFrame:
    """Convert PerformanceMetrics to display DataFrame."""
    data = {
        "Category": [],
        "Metric": [],
        "Value": [],
    }

    # Trade Statistics
    _add_metric(data, "Trade Statistics", "Total Trades", metrics.total_trades)
    _add_metric(data, "Trade Statistics", "Winning Trades", metrics.winning_trades)
    _add_metric(data, "Trade Statistics", "Losing Trades", metrics.losing_trades)
    _add_metric(data, "Trade Statistics", "Win Rate", f"{metrics.win_rate:.1%}")
    _add_metric(data, "Trade Statistics", "Max Consecutive Wins", metrics.max_consecutive_wins)
    _add_metric(data, "Trade Statistics", "Max Consecutive Losses", metrics.max_consecutive_losses)

    # P&L Metrics
    _add_metric(data, "Profit & Loss", "Total P&L", f"${metrics.total_pnl:,.2f}")
    _add_metric(data, "Profit & Loss", "Gross Profit", f"${metrics.gross_profit:,.2f}")
    _add_metric(data, "Profit & Loss", "Gross Loss", f"${metrics.gross_loss:,.2f}")
    _add_metric(data, "Profit & Loss", "Avg Trade P&L", f"${metrics.avg_trade_pnl:,.2f}")
    _add_metric(data, "Profit & Loss", "Avg Winning Trade", f"${metrics.avg_winning_trade:,.2f}")
    _add_metric(data, "Profit & Loss", "Avg Losing Trade", f"${metrics.avg_losing_trade:,.2f}")
    _add_metric(data, "Profit & Loss", "Largest Win", f"${metrics.largest_winning_trade:,.2f}")
    _add_metric(data, "Profit & Loss", "Largest Loss", f"${metrics.largest_losing_trade:,.2f}")

    # Risk Metrics
    _add_metric(data, "Risk Analysis", "Max Drawdown", f"${metrics.max_drawdown:,.2f}")
    _add_metric(data, "Risk Analysis", "Max Drawdown %", f"{metrics.max_drawdown_pct:.2%}")
    _add_metric(data, "Risk Analysis", "Max DD Duration", f"{metrics.max_drawdown_duration} days")
    _add_metric(data, "Risk Analysis", "VaR (95%)", f"${metrics.var_95:,.2f}")
    _add_metric(data, "Risk Analysis", "CVaR (95%)", f"${metrics.cvar_95:,.2f}")

    # Risk-Adjusted Returns
    _add_metric(data, "Risk-Adjusted", "Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
    _add_metric(data, "Risk-Adjusted", "Sortino Ratio", f"{metrics.sortino_ratio:.3f}")
    _add_metric(data, "Risk-Adjusted", "Profit Factor", f"{metrics.profit_factor:.2f}" if metrics.profit_factor < 100 else "∞")
    _add_metric(data, "Risk-Adjusted", "Payoff Ratio", f"{metrics.payoff_ratio:.2f}" if metrics.payoff_ratio < 100 else "∞")
    _add_metric(data, "Risk-Adjusted", "Expectancy", f"${metrics.expectancy:,.2f}")

    # Exposure & Timing
    _add_metric(data, "Exposure", "Time in Market", f"{metrics.total_exposure_time:.1%}")
    _add_metric(data, "Exposure", "Avg Holding Period", f"{metrics.avg_holding_period:.1f} days")
    _add_metric(data, "Exposure", "Max Holding Period", f"{metrics.max_holding_period:.0f} days")
    _add_metric(data, "Exposure", "Min Holding Period", f"{metrics.min_holding_period:.0f} days")

    # Annualized
    _add_metric(data, "Annualized", "Return", f"{metrics.annualized_return:.2%}")
    _add_metric(data, "Annualized", "Volatility", f"{metrics.annualized_volatility:.2%}")

    # Statistical
    _add_metric(data, "Distribution", "Skewness", f"{metrics.skewness:.3f}")
    _add_metric(data, "Distribution", "Kurtosis", f"{metrics.kurtosis:.3f}")

    return pd.DataFrame(data)


def _add_metric(data: dict, category: str, metric: str, value: object) -> None:
    """Helper to add metric to data dict."""
    data["Category"].append(category)
    data["Metric"].append(metric)
    data["Value"].append(value)


def run_monte_carlo_simulation(
    df_trades: pd.DataFrame,
    n_simulations: int = 1000,
    initial_capital: float = 100_000.0
) -> dict:
    """
    Run Monte Carlo simulation on trade sequence.

    Randomly shuffles trade order to estimate distribution of outcomes.

    Args:
        df_trades: DataFrame with trade records
        n_simulations: Number of simulations to run
        initial_capital: Starting capital

    Returns:
        Dict with simulation results including percentiles
    """
    if df_trades.empty or "pnl_usd" not in df_trades.columns:
        return {}

    # Get P&L from completed trades
    sells = df_trades[df_trades["position"] == "sell"]
    pnl_values = sells["pnl_usd"].values

    if len(pnl_values) == 0:
        return {}

    np.random.seed(42)

    final_pnls = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled_pnl = np.random.permutation(pnl_values)

        # Calculate cumulative P&L
        cum_pnl = np.cumsum(shuffled_pnl)
        portfolio = initial_capital + cum_pnl

        # Calculate max drawdown
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (running_max - portfolio) / running_max

        final_pnls.append(cum_pnl[-1])
        max_drawdowns.append(drawdown.max())

    return {
        "final_pnl_mean": np.mean(final_pnls),
        "final_pnl_std": np.std(final_pnls),
        "final_pnl_5th": np.percentile(final_pnls, 5),
        "final_pnl_25th": np.percentile(final_pnls, 25),
        "final_pnl_50th": np.percentile(final_pnls, 50),
        "final_pnl_75th": np.percentile(final_pnls, 75),
        "final_pnl_95th": np.percentile(final_pnls, 95),
        "max_dd_mean": np.mean(max_drawdowns),
        "max_dd_5th": np.percentile(max_drawdowns, 5),
        "max_dd_95th": np.percentile(max_drawdowns, 95),
        "simulations": n_simulations,
        "final_pnls": final_pnls,
        "max_drawdowns": max_drawdowns,
    }
