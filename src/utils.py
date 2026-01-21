"""
Utility functions for backtesting calculations.

This module provides functions for:
- P&L calculations (realized and mark-to-market)
- Performance metrics computation
- Strategy statistics
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
    position_open: bool
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Calculate realized and unrealized P&L for all trades.

    Matches buy/sell pairs and calculates profit/loss for each closed trade.
    If a position is still open, calculates mark-to-market adjustment.

    Args:
        df_trades: DataFrame containing trade records
        df_prices: DataFrame with price history
        commodity_chosen: Name of the traded commodity
        tons_conversion: Dictionary of tons conversion factors
        contract_size: Size of one contract
        position_open: Whether there's an open position

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
        ...     position_open=False
        ... )
    """
    df_trades = df_trades.copy()
    mtm_trade: Optional[Dict] = None

    if df_trades.empty:
        logger.warning("No trades to calculate P&L")
        return df_trades, mtm_trade

    # Initialize P&L column
    df_trades["pnl_usd"] = 0.0

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

        # Calculate P&L in USD
        buy_price_per_ton = buy_trade["trade_price"] * conversion_factor
        sell_price_per_ton = sell_trade["trade_price"] * conversion_factor

        pnl = (sell_price_per_ton - buy_price_per_ton) * contract_tons
        df_trades.at[sell_trade.name, "pnl_usd"] = pnl

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
