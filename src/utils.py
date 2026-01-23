"""
Utility functions for backtesting calculations.

This module provides functions for:
- P&L calculations (realized and mark-to-market)
- Performance metrics computation (basic and extended)
- Strategy statistics
- Transaction cost handling
"""

import logging
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR

# Configure module logger
logger = logging.getLogger(__name__)


def pnl_trades(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    commodity_chosen: str,
    tons_conversion: Dict[str, float],
    contract_size: float,
    position_open: bool,
    commission_per_trade: float = 0.0,
    slippage_pct: float = 0.0,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Calculate realized and unrealized P&L for all trades.

    Matches buy/sell pairs and calculates profit/loss for each closed trade.
    If a position is still open, calculates mark-to-market adjustment.
    Supports transaction costs (commission and slippage).

    Args:
        df_trades: DataFrame containing trade records
        df_prices: DataFrame with price history
        commodity_chosen: Name of the traded commodity
        tons_conversion: Dictionary of tons conversion factors
        contract_size: Size of one contract
        position_open: Whether there's an open position
        commission_per_trade: Fixed commission per trade (default: 0)
        slippage_pct: Slippage as percentage of price (default: 0)

    Returns:
        Tuple containing:
            - DataFrame with P&L columns added
            - MTM trade dict (or None if no open position)

    Example:
        >>> trades_with_pnl, mtm = pnl_trades(
        ...     df_trades=trades_df,
        ...     df_prices=prices_df,
        ...     commodity_chosen="Soybean",
        ...     tons_conversion=tons_conv,
        ...     contract_size=136.0,
        ...     position_open=False,
        ...     commission_per_trade=2.50,
        ...     slippage_pct=0.001
        ... )
    """
    df_trades = df_trades.copy()
    mtm_trade: Optional[Dict] = None

    if df_trades.empty:
        logger.warning("No trades to calculate P&L")
        return df_trades, mtm_trade

    # Initialize P&L columns
    df_trades["pnl_usd"] = 0.0
    df_trades["commission"] = 0.0
    df_trades["slippage"] = 0.0

    # Contract value in tons
    contract_tons = contract_size * tons_conversion[commodity_chosen]
    conversion_factor = tons_conversion[commodity_chosen]

    # Calculate P&L for completed trades (buy-sell pairs)
    for i in range(0, len(df_trades) - 1, 2):
        buy_trade = df_trades.iloc[i]
        sell_trade = df_trades.iloc[i + 1]

        # Validate trade pair
        if buy_trade["position"] != "buy" or sell_trade["position"] != "sell":
            logger.warning(f"Invalid trade pair at index {i}: expected buy-sell")
            continue

        # Calculate slippage (adverse price movement)
        buy_slippage = buy_trade["trade_price"] * slippage_pct
        sell_slippage = sell_trade["trade_price"] * slippage_pct

        # Adjusted prices after slippage
        buy_price_adj = buy_trade["trade_price"] + buy_slippage
        sell_price_adj = sell_trade["trade_price"] - sell_slippage

        # Calculate P&L in USD
        buy_price_per_ton = buy_price_adj * conversion_factor
        sell_price_per_ton = sell_price_adj * conversion_factor

        gross_pnl = (sell_price_per_ton - buy_price_per_ton) * contract_tons

        # Total commission for the round trip
        total_commission = commission_per_trade * 2
        total_slippage_cost = (buy_slippage + sell_slippage) * conversion_factor * contract_tons

        # Net P&L
        net_pnl = gross_pnl - total_commission

        df_trades.at[sell_trade.name, "pnl_usd"] = net_pnl
        df_trades.at[buy_trade.name, "commission"] = commission_per_trade
        df_trades.at[sell_trade.name, "commission"] = commission_per_trade
        df_trades.at[buy_trade.name, "slippage"] = buy_slippage * conversion_factor * contract_tons
        df_trades.at[sell_trade.name, "slippage"] = sell_slippage * conversion_factor * contract_tons

    # Calculate cumulative P&L
    df_trades["pnl_usd_cumsum"] = df_trades["pnl_usd"].cumsum()

    # Calculate MTM for open position
    if position_open and not df_trades.empty:
        last_trade = df_trades.iloc[-1]

        if last_trade["position"] == "buy":
            last_market_price = df_prices[commodity_chosen].iloc[-1]

            buy_price_per_ton = last_trade["trade_price"] * conversion_factor
            market_price_per_ton = last_market_price * conversion_factor

            mtm_pnl = (market_price_per_ton - buy_price_per_ton) * contract_tons

            # Deduct estimated exit costs
            mtm_pnl -= commission_per_trade
            mtm_pnl -= (last_market_price * slippage_pct * conversion_factor * contract_tons)

            mtm_trade = {
                "date": df_prices.index[-1],
                "pnl_usd": mtm_pnl
            }

            logger.info(f"Open position MTM: ${mtm_pnl:,.2f}")

    return df_trades, mtm_trade


def backtest_performance(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: Optional[Dict] = None,
    contract_size: Optional[float] = None,
    tons_conversion: Optional[Dict[str, float]] = None,
    commodity_chosen: Optional[str] = None,
    position_open: Optional[bool] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics for a backtest.

    Args:
        df_trades: DataFrame with trade records and P&L
        df_prices: DataFrame with price history
        mtm_trade: Mark-to-market adjustment dict (optional)
        contract_size: Size of one contract
        tons_conversion: Dictionary of tons conversion factors
        commodity_chosen: Name of the traded commodity
        position_open: Whether there's an open position

    Returns:
        DataFrame with performance metrics and their values

    Metrics calculated:
        - Total Buys/Sells
        - Complete Trades
        - Open Positions
        - Realized/MTM/Total Profit
        - Win Rate
        - Max Drawdown
        - Sharpe Ratio
        - Best/Worst Trade
        - Mean Trade Duration
        - Gross Exposure
        - VaR (95%)
    """
    # Basic trade counts
    total_buys = (df_trades["position"] == "buy").sum()
    total_sells = (df_trades["position"] == "sell").sum()
    complete_trades = min(total_buys, total_sells)
    open_positions = total_buys - total_sells

    # P&L calculations
    realized_profit = df_trades["pnl_usd"].sum()
    mtm_profit = mtm_trade["pnl_usd"] if mtm_trade else 0.0
    total_profit = realized_profit + mtm_profit

    # Win rate (only for completed trades)
    if complete_trades > 0:
        # P&L is recorded on sell trades
        sell_trades_pnl = df_trades[df_trades["position"] == "sell"]["pnl_usd"]
        win_trades = (sell_trades_pnl > 0).sum()
        win_rate = win_trades / complete_trades
    else:
        win_rate = 0.0

    # Drawdown calculation
    if "pnl_usd_cumsum" in df_trades.columns and not df_trades["pnl_usd_cumsum"].empty:
        cumulative_pnl = df_trades["pnl_usd_cumsum"]
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
    else:
        max_drawdown = 0.0

    # Sharpe Ratio (annualized)
    pnl_series = df_trades["pnl_usd"].dropna()
    if len(pnl_series) > 1 and pnl_series.std() != 0:
        sharpe_ratio = (pnl_series.mean() / pnl_series.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe_ratio = np.nan

    # Best and worst trades
    best_trade = df_trades["pnl_usd"].max() if not df_trades.empty else 0.0
    worst_trade = df_trades["pnl_usd"].min() if not df_trades.empty else 0.0

    # Mean trade duration
    if len(df_trades) >= 2:
        trade_dates = pd.to_datetime(df_trades.index)
        durations = trade_dates.to_series().diff().dropna()
        mean_duration = durations.mean().days if not durations.empty else np.nan
    else:
        mean_duration = np.nan

    # Backtest duration
    if not df_prices.empty:
        backtest_duration = (df_prices.index[-1] - df_prices.index[0]).days
    else:
        backtest_duration = 0

    # Gross exposure calculation
    gross_exposure = 0.0
    if (position_open and not df_trades.empty and
            commodity_chosen and tons_conversion and contract_size):
        if df_trades.iloc[-1]["position"] == "buy":
            last_price = df_prices[commodity_chosen].iloc[-1]
            gross_exposure = (
                contract_size *
                tons_conversion[commodity_chosen] *
                last_price
            )

    # VaR calculation (Historical 95%)
    var_95 = 0.0
    if commodity_chosen and not df_prices.empty and gross_exposure > 0:
        returns = df_prices[commodity_chosen].pct_change().dropna()
        if not returns.empty:
            var_95 = np.percentile(returns, 5) * gross_exposure

    # Build performance summary DataFrame
    metrics = pd.DataFrame({
        "Metric": [
            "Total Buys",
            "Total Sells",
            "Complete Trades",
            "Open Positions",
            "Realized Profit (USD)",
            "MTM Adjustment (USD)",
            "Total Profit (USD)",
            "Win Rate (%)",
            "Max Drawdown (USD)",
            "Sharpe Ratio",
            "Best Trade (USD)",
            "Worst Trade (USD)",
            "Mean Trade Duration (days)",
            "Backtest Duration (days)",
            "Gross Exposure (USD)",
            "VaR 95% (Historical - USD)"
        ],
        "Value": [
            total_buys,
            total_sells,
            complete_trades,
            open_positions,
            realized_profit,
            mtm_profit,
            total_profit,
            win_rate * 100,
            max_drawdown,
            sharpe_ratio,
            best_trade,
            worst_trade,
            mean_duration,
            backtest_duration,
            gross_exposure,
            var_95
        ]
    })

    return metrics.round(2)


def backtest_performance_extended(
    df_trades: pd.DataFrame,
    df_prices: pd.DataFrame,
    mtm_trade: Optional[Dict] = None,
    contract_size: Optional[float] = None,
    tons_conversion: Optional[Dict[str, float]] = None,
    commodity_chosen: Optional[str] = None,
    position_open: Optional[bool] = None
) -> pd.DataFrame:
    """
    Calculate extended performance metrics including advanced risk measures.

    This function extends the basic performance metrics with:
    - Sortino Ratio (downside risk adjusted)
    - Calmar Ratio (return vs max drawdown)
    - Profit Factor (gross profit / gross loss)
    - Average Win/Loss amounts
    - Consecutive wins/losses
    - Recovery Factor

    Args:
        df_trades: DataFrame with trade records and P&L
        df_prices: DataFrame with price history
        mtm_trade: Mark-to-market adjustment dict (optional)
        contract_size: Size of one contract
        tons_conversion: Dictionary of tons conversion factors
        commodity_chosen: Name of the traded commodity
        position_open: Whether there's an open position

    Returns:
        DataFrame with extended performance metrics
    """
    # Get basic metrics first
    basic_metrics = backtest_performance(
        df_trades, df_prices, mtm_trade, contract_size,
        tons_conversion, commodity_chosen, position_open
    )

    # Extract needed values
    sell_trades = df_trades[df_trades["position"] == "sell"]
    pnl_series = sell_trades["pnl_usd"].dropna()

    # Sortino Ratio (using downside deviation)
    if len(pnl_series) > 1:
        downside_returns = pnl_series[pnl_series < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino_ratio = (pnl_series.mean() / downside_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino_ratio = np.nan
    else:
        sortino_ratio = np.nan

    # Profit Factor (gross profit / gross loss)
    winning_trades = pnl_series[pnl_series > 0]
    losing_trades = pnl_series[pnl_series < 0]

    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else np.nan

    # Average win and average loss
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

    # Win/Loss statistics
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)

    # Consecutive wins/losses
    if len(pnl_series) > 0:
        is_win = (pnl_series > 0).astype(int)
        is_loss = (pnl_series < 0).astype(int)

        # Calculate consecutive runs
        win_groups = (is_win != is_win.shift()).cumsum()
        loss_groups = (is_loss != is_loss.shift()).cumsum()

        max_consecutive_wins = is_win.groupby(win_groups).sum().max()
        max_consecutive_losses = is_loss.groupby(loss_groups).sum().max()
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    # Calmar Ratio (annualized return / max drawdown)
    total_profit = basic_metrics[basic_metrics["Metric"] == "Total Profit (USD)"]["Value"].values[0]
    max_drawdown = basic_metrics[basic_metrics["Metric"] == "Max Drawdown (USD)"]["Value"].values[0]
    backtest_duration = basic_metrics[basic_metrics["Metric"] == "Backtest Duration (days)"]["Value"].values[0]

    if max_drawdown > 0 and backtest_duration > 0:
        annualized_return = (total_profit / backtest_duration) * 365
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = np.nan

    # Recovery Factor (total profit / max drawdown)
    if max_drawdown > 0:
        recovery_factor = total_profit / max_drawdown
    else:
        recovery_factor = np.nan

    # Expectancy (average profit per trade)
    if len(pnl_series) > 0:
        expectancy = pnl_series.mean()
    else:
        expectancy = 0

    # Total commissions and slippage (if tracked)
    total_commission = df_trades["commission"].sum() if "commission" in df_trades.columns else 0
    total_slippage = df_trades["slippage"].sum() if "slippage" in df_trades.columns else 0

    # Add extended metrics
    extended_metrics = pd.DataFrame({
        "Metric": [
            "Sortino Ratio",
            "Calmar Ratio",
            "Profit Factor",
            "Average Win (USD)",
            "Average Loss (USD)",
            "Number of Wins",
            "Number of Losses",
            "Max Consecutive Wins",
            "Max Consecutive Losses",
            "Recovery Factor",
            "Expectancy (USD/trade)",
            "Total Commission (USD)",
            "Total Slippage Cost (USD)",
        ],
        "Value": [
            sortino_ratio,
            calmar_ratio,
            profit_factor,
            avg_win,
            avg_loss,
            num_wins,
            num_losses,
            max_consecutive_wins,
            max_consecutive_losses,
            recovery_factor,
            expectancy,
            total_commission,
            total_slippage,
        ]
    })

    # Combine basic and extended metrics
    all_metrics = pd.concat([basic_metrics, extended_metrics], ignore_index=True)

    return all_metrics.round(2)


def strategy_describe(
    df: pd.DataFrame,
    tons_conversion: Dict[str, float],
    backtest_strategy: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for commodity ratios.

    Generates statistics for all possible commodity pair ratios,
    useful for identifying trading opportunities.

    Args:
        df: DataFrame with commodity prices
        tons_conversion: Dictionary of tons conversion factors
        backtest_strategy: Strategy type (only 'ratio' is supported)

    Returns:
        DataFrame with ratio statistics including:
        - count, mean, std, min, 25%, 50%, 75%, max
        - coefficient of variation

    Example:
        >>> stats = strategy_describe(
        ...     df=prices_df,
        ...     tons_conversion=tons_conv,
        ...     backtest_strategy="ratio"
        ... )
    """
    if backtest_strategy != "ratio":
        logger.info("Strategy describe only available for ratio strategy")
        return pd.DataFrame(index=df.index)

    summary_list: List[pd.Series] = []

    # Calculate statistics for all commodity pairs
    for col1, col2 in permutations(df.columns, 2):
        if col1 in tons_conversion and col2 in tons_conversion:
            ratio_name = f"{col1}/{col2}"

            # Calculate ratio in metric ton terms
            ratio_series = (
                (df[col1] * tons_conversion[col1]) /
                (df[col2] * tons_conversion[col2])
            )

            # Get descriptive statistics
            stats = ratio_series.describe()
            stats["coefficient variation"] = stats["std"] / stats["mean"]
            stats.name = ratio_name
            summary_list.append(stats)

    if not summary_list:
        logger.warning("No valid commodity pairs found for ratio calculation")
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_list)
    return summary_df.round(4)


def calculate_drawdown_series(cumulative_pnl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate drawdown and drawdown percentage series.

    Args:
        cumulative_pnl: Series of cumulative P&L values

    Returns:
        Tuple of (drawdown_series, drawdown_pct_series)
    """
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl

    # Calculate percentage drawdown (avoiding division by zero)
    drawdown_pct = pd.Series(index=cumulative_pnl.index, dtype=float)
    mask = running_max > 0
    drawdown_pct[mask] = (drawdown[mask] / running_max[mask]) * 100
    drawdown_pct[~mask] = 0

    return drawdown, drawdown_pct


def calculate_monthly_returns(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from trade data.

    Args:
        df_trades: DataFrame with trade records and P&L

    Returns:
        DataFrame with monthly return statistics
    """
    if df_trades.empty or "pnl_usd" not in df_trades.columns:
        return pd.DataFrame()

    df_trades_copy = df_trades.copy()
    df_trades_copy.index = pd.to_datetime(df_trades_copy.index)

    # Group by month
    monthly_pnl = df_trades_copy["pnl_usd"].groupby(
        df_trades_copy.index.to_period("M")
    ).sum()

    monthly_stats = pd.DataFrame({
        "Period": monthly_pnl.index.astype(str),
        "P&L (USD)": monthly_pnl.values,
        "Cumulative P&L": monthly_pnl.cumsum().values
    })

    return monthly_stats


def calculate_risk_metrics(
    df_prices: pd.DataFrame,
    commodity_chosen: str,
    contract_size: float,
    tons_conversion: Dict[str, float],
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Calculate various risk metrics for a commodity position.

    Args:
        df_prices: DataFrame with price history
        commodity_chosen: Name of the commodity
        contract_size: Size of one contract
        tons_conversion: Dictionary of tons conversion factors
        confidence_levels: List of confidence levels for VaR calculation

    Returns:
        Dictionary of risk metrics
    """
    if commodity_chosen not in df_prices.columns:
        return {}

    prices = df_prices[commodity_chosen]
    returns = prices.pct_change().dropna()

    # Position value
    contract_value = contract_size * tons_conversion[commodity_chosen] * prices.iloc[-1]

    metrics = {
        "Daily Volatility (%)": returns.std() * 100,
        "Annualized Volatility (%)": returns.std() * np.sqrt(252) * 100,
        "Contract Value (USD)": contract_value,
    }

    # Historical VaR at different confidence levels
    for conf in confidence_levels:
        percentile = (1 - conf) * 100
        var_pct = np.percentile(returns, percentile)
        var_usd = var_pct * contract_value
        metrics[f"VaR {int(conf*100)}% (USD)"] = var_usd

    # Conditional VaR (Expected Shortfall)
    for conf in confidence_levels:
        percentile = (1 - conf) * 100
        var_threshold = np.percentile(returns, percentile)
        cvar = returns[returns <= var_threshold].mean()
        cvar_usd = cvar * contract_value if not np.isnan(cvar) else np.nan
        metrics[f"CVaR {int(conf*100)}% (USD)"] = cvar_usd

    return metrics
