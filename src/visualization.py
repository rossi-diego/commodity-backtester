"""
Visualization module for backtesting results.

This module provides functions to create interactive charts using Plotly,
displaying price movements, trading signals, and P&L evolution.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
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


# Chart color scheme (professional palette)
COLORS = {
    "price_line": "#2E86AB",
    "buy_signal": "#28A745",
    "sell_signal": "#DC3545",
    "mtm_marker": "#FFC107",
    "ratio_line": "#6F42C1",
    "entry_level": "#28A745",
    "exit_level": "#DC3545",
    "pnl_line": "#2E86AB",
    "pnl_positive": "#28A745",
    "pnl_negative": "#DC3545",
    "drawdown_fill": "rgba(220, 53, 69, 0.3)",
    "grid": "#E9ECEF",
    "background": "#FFFFFF",
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
    use_streamlit: bool = True,
    chart_type: str = "all"
) -> None:
    """
    Generate interactive visualization charts for backtest results.

    Creates charts based on chart_type:
    - "all": All three charts (price, ratio, P&L)
    - "price": Price chart with buy/sell signals
    - "ratio": Ratio behavior with entry/exit thresholds
    - "pnl": Cumulative P&L evolution

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
        chart_type: Type of chart to display ("all", "price", "ratio", "pnl")

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
        ...     tons_conversion=tons_conv,
        ...     chart_type="price"
        ... )
    """
    # Prepare price data for plotting
    df_plot = df_prices.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot = df_plot.loc[str(start_date):str(end_date)]

    if chart_type in ["all", "price"]:
        # Create price chart with signals
        fig_price = _create_price_chart(
            df_plot=df_plot,
            df_trades=df_trades,
            commodity_chosen=commodity_chosen,
            mtm_trade=mtm_trade
        )
        _render_chart(fig_price, use_streamlit)

    if chart_type in ["all", "ratio"]:
        # Create ratio chart (if ratio column exists)
        if "ratio" in df_plot.columns:
            fig_ratio = _create_ratio_chart(
                df_plot=df_plot,
                down_entry=down_entry,
                up_exit=up_exit
            )
            _render_chart(fig_ratio, use_streamlit)
        else:
            if chart_type == "ratio":
                logger.warning("Ratio column not found in DataFrame - skipping ratio chart")

    if chart_type in ["all", "pnl"]:
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
        line=dict(color=COLORS["price_line"], width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:,.2f}<extra></extra>"
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
                    size=14,
                    color=COLORS["buy_signal"],
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>BUY</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index,
                y=sells["trade_price"],
                mode="markers",
                name="Sell Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color=COLORS["sell_signal"],
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>SELL</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:,.2f}<extra></extra>"
            ))

    # MTM position marker
    if mtm_trade and mtm_trade["date"] in df_plot.index:
        last_price = df_plot[commodity_chosen].loc[mtm_trade["date"]]
        pnl_color = COLORS["pnl_positive"] if mtm_trade["pnl_usd"] >= 0 else COLORS["pnl_negative"]
        fig.add_trace(go.Scatter(
            x=[mtm_trade["date"]],
            y=[last_price],
            mode="markers",
            name="Open Position (MTM)",
            marker=dict(
                symbol="circle",
                size=16,
                color=pnl_color,
                line=dict(color="white", width=2)
            ),
            hovertemplate=f"<b>OPEN POSITION</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.2f}}<br>MTM P&L: ${mtm_trade['pnl_usd']:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{commodity_chosen} Price with Trading Signals</b>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title=f"Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickformat="$,.0f"
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

    # Calculate mean for reference
    ratio_mean = df_plot["ratio"].mean()

    # Ratio line
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["ratio"],
        mode="lines",
        name="Ratio",
        line=dict(color=COLORS["ratio_line"], width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ratio: %{y:.4f}<extra></extra>"
    ))

    # Entry threshold
    fig.add_hline(
        y=down_entry,
        line_dash="dash",
        line_color=COLORS["entry_level"],
        line_width=2,
        annotation_text=f"Entry: {down_entry:.4f}",
        annotation_position="bottom left",
        annotation_font_color=COLORS["entry_level"]
    )

    # Exit threshold
    fig.add_hline(
        y=up_exit,
        line_dash="dash",
        line_color=COLORS["exit_level"],
        line_width=2,
        annotation_text=f"Exit: {up_exit:.4f}",
        annotation_position="top left",
        annotation_font_color=COLORS["exit_level"]
    )

    # Mean line
    fig.add_hline(
        y=ratio_mean,
        line_dash="dot",
        line_color="#6C757D",
        line_width=1,
        annotation_text=f"Mean: {ratio_mean:.4f}",
        annotation_position="right",
        annotation_font_color="#6C757D"
    )

    # Add shaded regions
    fig.add_hrect(
        y0=down_entry, y1=df_plot["ratio"].min() * 0.95,
        fillcolor="rgba(40, 167, 69, 0.1)",
        layer="below",
        line_width=0
    )

    fig.add_hrect(
        y0=up_exit, y1=df_plot["ratio"].max() * 1.05,
        fillcolor="rgba(220, 53, 69, 0.1)",
        layer="below",
        line_width=0
    )

    fig.update_layout(
        title=dict(
            text="<b>Price Ratio Behavior</b>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Ratio (Metric Tons)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"]
        )
    )

    return fig


def _create_pnl_chart(
    df_trades: pd.DataFrame,
    mtm_trade: Optional[Dict]
) -> go.Figure:
    """Create cumulative P&L chart with drawdown visualization."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("<b>Cumulative P&L</b>", "<b>Drawdown</b>")
    )

    cumulative_pnl = df_trades["pnl_usd_cumsum"]

    # Color the P&L line based on whether it's positive or negative
    colors = [COLORS["pnl_positive"] if v >= 0 else COLORS["pnl_negative"] for v in cumulative_pnl]

    # Cumulative P&L line
    fig.add_trace(
        go.Scatter(
            x=df_trades.index,
            y=cumulative_pnl,
            mode="lines",
            name="Realized P&L",
            line=dict(color=COLORS["pnl_line"], width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.2)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Cumulative P&L: $%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#6C757D", line_width=1, row=1, col=1)

    # MTM adjustment point
    if mtm_trade and not df_trades.empty:
        final_pnl = cumulative_pnl.iloc[-1]
        mtm_point = final_pnl + mtm_trade["pnl_usd"]
        pnl_color = COLORS["pnl_positive"] if mtm_trade["pnl_usd"] >= 0 else COLORS["pnl_negative"]

        fig.add_trace(
            go.Scatter(
                x=[mtm_trade["date"]],
                y=[mtm_point],
                mode="markers",
                name="MTM Adjustment",
                marker=dict(
                    color=pnl_color,
                    size=14,
                    line=dict(color="white", width=2),
                    symbol="star"
                ),
                hovertemplate=f"<b>MTM ADJUSTMENT</b><br>Total P&L: ${mtm_point:,.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Add connecting line to MTM point
        fig.add_trace(
            go.Scatter(
                x=[df_trades.index[-1], mtm_trade["date"]],
                y=[final_pnl, mtm_point],
                mode="lines",
                name="MTM Line",
                line=dict(color=pnl_color, width=2, dash="dot"),
                showlegend=False
            ),
            row=1, col=1
        )

    # Calculate and plot drawdown
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl

    fig.add_trace(
        go.Scatter(
            x=df_trades.index,
            y=-drawdown,  # Negative to show drawdown going down
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["sell_signal"], width=1),
            fill="tozeroy",
            fillcolor=COLORS["drawdown_fill"],
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drawdown: $%{y:,.2f}<extra></extra>"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        height=500
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="P&L (USD)",
        tickformat="$,.0f",
        showgrid=True,
        gridcolor=COLORS["grid"],
        row=1, col=1
    )

    fig.update_yaxes(
        title_text="Drawdown (USD)",
        tickformat="$,.0f",
        showgrid=True,
        gridcolor=COLORS["grid"],
        row=2, col=1
    )

    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor=COLORS["grid"],
        row=2, col=1
    )

    return fig


def create_strategy_comparison_chart(
    results: List[Dict[str, Any]],
    metric: str = "cumulative_pnl"
) -> go.Figure:
    """
    Create a comparison chart for multiple strategy backtests.

    Args:
        results: List of backtest result dictionaries, each containing:
            - strategy_name: Name of the strategy
            - df_trades: DataFrame with trade records
            - metrics: Performance metrics DataFrame
        metric: Metric to compare ("cumulative_pnl", "win_rate", "sharpe")

    Returns:
        Plotly figure with comparison chart
    """
    fig = go.Figure()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        strategy_name = result.get("strategy_name", f"Strategy {idx + 1}")
        df_trades = result.get("df_trades", pd.DataFrame())

        if not df_trades.empty and "pnl_usd_cumsum" in df_trades.columns:
            fig.add_trace(go.Scatter(
                x=df_trades.index,
                y=df_trades["pnl_usd_cumsum"],
                mode="lines",
                name=strategy_name,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{strategy_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>P&L: $%{{y:,.2f}}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(
            text="<b>Strategy Comparison - Cumulative P&L</b>",
            font=dict(size=18)
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (USD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        yaxis=dict(tickformat="$,.0f")
    )

    return fig


def create_equity_curve(
    df_trades: pd.DataFrame,
    initial_capital: float = 100000,
    title: str = "Equity Curve"
) -> go.Figure:
    """
    Create an equity curve chart showing portfolio value over time.

    Args:
        df_trades: DataFrame with trade records and P&L
        initial_capital: Starting capital amount
        title: Chart title

    Returns:
        Plotly figure with equity curve
    """
    if df_trades.empty or "pnl_usd_cumsum" not in df_trades.columns:
        logger.warning("No trade data available for equity curve")
        return go.Figure()

    equity = initial_capital + df_trades["pnl_usd_cumsum"]

    fig = go.Figure()

    # Equity line
    fig.add_trace(go.Scatter(
        x=df_trades.index,
        y=equity,
        mode="lines",
        name="Portfolio Value",
        line=dict(color=COLORS["pnl_line"], width=2),
        fill="tozeroy",
        fillcolor="rgba(46, 134, 171, 0.2)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Portfolio: $%{y:,.2f}<extra></extra>"
    ))

    # Initial capital line
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="#6C757D",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(tickformat="$,.0f"),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def create_trade_distribution(df_trades: pd.DataFrame) -> go.Figure:
    """
    Create a histogram showing the distribution of trade P&L.

    Args:
        df_trades: DataFrame with trade records and P&L

    Returns:
        Plotly figure with P&L distribution histogram
    """
    if df_trades.empty or "pnl_usd" not in df_trades.columns:
        return go.Figure()

    # Get P&L for completed trades (sell trades)
    sell_pnl = df_trades[df_trades["position"] == "sell"]["pnl_usd"]

    if sell_pnl.empty:
        return go.Figure()

    fig = go.Figure()

    # Color based on win/loss
    colors = [COLORS["pnl_positive"] if x >= 0 else COLORS["pnl_negative"] for x in sell_pnl]

    fig.add_trace(go.Histogram(
        x=sell_pnl,
        name="Trade P&L",
        marker_color=COLORS["pnl_line"],
        opacity=0.75,
        hovertemplate="P&L Range: $%{x}<br>Count: %{y}<extra></extra>"
    ))

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#6C757D",
        annotation_text="Breakeven",
        annotation_position="top"
    )

    # Add mean line
    mean_pnl = sell_pnl.mean()
    fig.add_vline(
        x=mean_pnl,
        line_dash="dot",
        line_color=COLORS["pnl_positive"] if mean_pnl >= 0 else COLORS["pnl_negative"],
        annotation_text=f"Mean: ${mean_pnl:,.0f}",
        annotation_position="bottom"
    )

    fig.update_layout(
        title=dict(text="<b>Trade P&L Distribution</b>", font=dict(size=18)),
        xaxis_title="P&L per Trade (USD)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(tickformat="$,.0f")
    )

    return fig


def create_monthly_returns_heatmap(df_trades: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing monthly returns.

    Args:
        df_trades: DataFrame with trade records and P&L

    Returns:
        Plotly figure with monthly returns heatmap
    """
    if df_trades.empty or "pnl_usd" not in df_trades.columns:
        return go.Figure()

    df_trades_copy = df_trades.copy()
    df_trades_copy.index = pd.to_datetime(df_trades_copy.index)

    # Calculate monthly P&L
    df_trades_copy["year"] = df_trades_copy.index.year
    df_trades_copy["month"] = df_trades_copy.index.month

    monthly_pnl = df_trades_copy.groupby(["year", "month"])["pnl_usd"].sum().unstack()

    if monthly_pnl.empty:
        return go.Figure()

    # Month names for labels
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=monthly_pnl.values,
        x=[month_names[m - 1] for m in monthly_pnl.columns],
        y=monthly_pnl.index,
        colorscale=[[0, COLORS["pnl_negative"]], [0.5, "#FFFFFF"], [1, COLORS["pnl_positive"]]],
        zmid=0,
        text=[[f"${v:,.0f}" if not np.isnan(v) else "" for v in row] for row in monthly_pnl.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>P&L: $%{z:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Monthly Returns Heatmap</b>", font=dict(size=18)),
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def _render_chart(fig: go.Figure, use_streamlit: bool) -> None:
    """Render chart in Streamlit or standalone."""
    if use_streamlit and HAS_STREAMLIT and st is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig.show()
