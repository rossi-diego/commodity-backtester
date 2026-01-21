"""
Visualization module for backtesting results.

This module provides functions to create interactive charts using Plotly,
displaying price movements, trading signals, and P&L evolution.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure module logger
logger = logging.getLogger(__name__)

# Optional Streamlit import
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False


# Chart color scheme
COLORS = {
    "price_line": "#1f77b4",
    "buy_signal": "#2ca02c",
    "sell_signal": "#d62728",
    "mtm_marker": "#ff7f0e",
    "ratio_line": "#9467bd",
    "entry_level": "#2ca02c",
    "exit_level": "#d62728",
    "pnl_line": "#1f77b4",
}


def backtest_charts(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    commodity_chosen: str,
    down_entry: float,
    up_exit: float,
    start_date: Any,
    end_date: Any,
    mtm_trade: Optional[Dict],
    tons_conversion: Dict[str, float],
    use_streamlit: bool = True
) -> None:
    """
    Generate interactive visualization charts for backtest results.

    Creates three charts:
    1. Price chart with buy/sell signals
    2. Ratio behavior with entry/exit thresholds
    3. Cumulative P&L evolution

    Args:
        df_prices: DataFrame with commodity price history
        df_trades: DataFrame with executed trades
        commodity_chosen: Name of the traded commodity
        down_entry: Entry threshold for ratio strategy
        up_exit: Exit threshold for ratio strategy
        start_date: Backtest start date
        end_date: Backtest end date
        mtm_trade: Mark-to-market trade dict (optional)
        tons_conversion: Dictionary of tons conversion factors
        use_streamlit: Whether to render in Streamlit (default: True)

    Example:
        >>> backtest_charts(
        ...     df_prices=prices_df,
        ...     df_trades=trades_df,
        ...     commodity_chosen="Soybean",
        ...     down_entry=0.95,
        ...     up_exit=1.05,
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     mtm_trade=None,
        ...     tons_conversion=tons_conv
        ... )
    """
    # Prepare price data for plotting
    df_plot = df_prices.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot = df_plot.loc[str(start_date):str(end_date)]

    # Create price chart with signals
    fig_price = _create_price_chart(
        df_plot=df_plot,
        df_trades=df_trades,
        commodity_chosen=commodity_chosen,
        mtm_trade=mtm_trade
    )

    _render_chart(fig_price, use_streamlit)

    # Create ratio chart (if ratio column exists)
    if "ratio" in df_plot.columns:
        fig_ratio = _create_ratio_chart(
            df_plot=df_plot,
            down_entry=down_entry,
            up_exit=up_exit
        )
        _render_chart(fig_ratio, use_streamlit)
    else:
        logger.warning("Ratio column not found in DataFrame - skipping ratio chart")

    # Create P&L chart
    if not df_trades.empty and "pnl_usd_cumsum" in df_trades.columns:
        fig_pnl = _create_pnl_chart(
            df_trades=df_trades,
            mtm_trade=mtm_trade
        )
        _render_chart(fig_pnl, use_streamlit)


def _create_price_chart(
    df_plot: pd.DataFrame,
    df_trades: pd.DataFrame,
    commodity_chosen: str,
    mtm_trade: Optional[Dict]
) -> go.Figure:
    """Create price chart with trading signals."""
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[commodity_chosen],
        mode="lines",
        name=f"{commodity_chosen} Price",
        line=dict(color=COLORS["price_line"], width=1.5)
    ))

    # Buy signals
    if not df_trades.empty:
        buys = df_trades[df_trades["position"] == "buy"]
        sells = df_trades[df_trades["position"] == "sell"]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index,
                y=buys["trade_price"],
                mode="markers",
                name="Buy Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=COLORS["buy_signal"],
                    line=dict(color="black", width=1)
                )
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index,
                y=sells["trade_price"],
                mode="markers",
                name="Sell Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=COLORS["sell_signal"],
                    line=dict(color="black", width=1)
                )
            ))

    # MTM position marker
    if mtm_trade and mtm_trade["date"] in df_plot.index:
        last_price = df_plot[commodity_chosen].loc[mtm_trade["date"]]
        fig.add_trace(go.Scatter(
            x=[mtm_trade["date"]],
            y=[last_price],
            mode="markers",
            name="Open MTM Position",
            marker=dict(
                symbol="circle",
                size=14,
                color=COLORS["mtm_marker"],
                line=dict(color="black", width=2)
            )
        ))

    fig.update_layout(
        title="Price Chart with Trading Signals",
        xaxis_title="Date",
        yaxis_title=f"{commodity_chosen} Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def _create_ratio_chart(
    df_plot: pd.DataFrame,
    down_entry: float,
    up_exit: float
) -> go.Figure:
    """Create ratio chart with entry/exit thresholds."""
    fig = go.Figure()

    # Ratio line
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["ratio"],
        mode="lines",
        name="Ratio",
        line=dict(color=COLORS["ratio_line"], width=1.5)
    ))

    # Entry threshold
    fig.add_hline(
        y=down_entry,
        line_dash="dash",
        line_color=COLORS["entry_level"],
        annotation_text=f"Entry: {down_entry:.2f}",
        annotation_position="bottom left"
    )

    # Exit threshold
    fig.add_hline(
        y=up_exit,
        line_dash="dash",
        line_color=COLORS["exit_level"],
        annotation_text=f"Exit: {up_exit:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Ratio Behavior",
        xaxis_title="Date",
        yaxis_title="Ratio (Metric Tons)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def _create_pnl_chart(
    df_trades: pd.DataFrame,
    mtm_trade: Optional[Dict]
) -> go.Figure:
    """Create cumulative P&L chart."""
    fig = go.Figure()

    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=df_trades.index,
        y=df_trades["pnl_usd_cumsum"],
        mode="lines",
        name="Realized P&L",
        line=dict(color=COLORS["pnl_line"], width=2)
    ))

    # MTM adjustment point
    if mtm_trade and not df_trades.empty:
        mtm_point = df_trades["pnl_usd_cumsum"].iloc[-1] + mtm_trade["pnl_usd"]
        fig.add_trace(go.Scatter(
            x=[mtm_trade["date"]],
            y=[mtm_point],
            mode="markers",
            name="MTM Adjustment",
            marker=dict(
                color=COLORS["mtm_marker"],
                size=14,
                line=dict(color="black", width=2)
            )
        ))

    fig.update_layout(
        title="Cumulative P&L",
        xaxis_title="Date",
        yaxis_title="P&L (USD)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def _render_chart(fig: go.Figure, use_streamlit: bool) -> None:
    """Render chart in Streamlit or standalone."""
    if use_streamlit and HAS_STREAMLIT and st is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    else:
        fig.show()
