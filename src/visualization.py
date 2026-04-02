"""Plotly chart builders for the backtest results."""

from __future__ import annotations

import datetime
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False


__all__ = ["backtest_charts", "create_strategy_comparison_chart"]

ChartType = Literal["price", "ratio", "pnl"]


def backtest_charts(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    commodity_chosen: str,
    down_entry: float,
    up_exit: float,
    start_date: datetime.date | str,
    end_date: datetime.date | str,
    mtm_trade: dict[str, object] | None,
    tons_conversion: dict[str, float],
    use_streamlit: bool = True,
    chart_type: ChartType = "price",
) -> None:
    """Render a single interactive Plotly chart based on ``chart_type``.

    Parameters
    ----------
    chart_type:
        ``"price"`` — price series with buy/sell signals.
        ``"ratio"`` — spread ratio with entry/exit threshold lines.
        ``"pnl"``   — cumulative realised PnL curve.
    """
    plot_df = df_prices.copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    plot_df = plot_df.loc[str(start_date) : str(end_date)]  # type: ignore[misc]

    _render = _streamlit_render if (use_streamlit and _HAS_STREAMLIT) else _standalone_render

    if chart_type == "price":
        _render(_price_chart(plot_df, df_trades, commodity_chosen, mtm_trade))

    elif chart_type == "ratio":
        if "ratio" not in plot_df.columns:
            if _HAS_STREAMLIT and st is not None:
                st.info("Ratio chart is only available for the Ratio strategy.")
            return
        _render(_ratio_chart(plot_df, down_entry, up_exit))

    elif chart_type == "pnl":
        if "pnl_usd_cumsum" not in df_trades.columns:
            return
        _render(_pnl_chart(df_trades, mtm_trade))


def create_strategy_comparison_chart(
    results: dict[str, pd.DataFrame],
    use_streamlit: bool = True,
) -> None:
    """Overlay cumulative PnL curves for multiple strategies.

    Parameters
    ----------
    results:
        Mapping of strategy name → trade log DataFrame (must have ``pnl_usd_cumsum``).
    """
    fig = go.Figure()
    colours = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b"]

    for i, (name, df_trades) in enumerate(results.items()):
        if "pnl_usd_cumsum" not in df_trades.columns or df_trades.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df_trades.index,
                y=df_trades["pnl_usd_cumsum"],
                mode="lines",
                name=name,
                line={"color": colours[i % len(colours)]},
            )
        )

    fig.update_layout(
        title="Strategy Comparison — Cumulative PnL (USD)",
        xaxis_title="Date",
        yaxis_title="PnL (USD)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )

    _render = _streamlit_render if (use_streamlit and _HAS_STREAMLIT) else _standalone_render
    _render(fig)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def _price_chart(
    plot_df: pd.DataFrame,
    df_trades: pd.DataFrame,
    commodity_chosen: str,
    mtm_trade: dict[str, object] | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[commodity_chosen],
            mode="lines",
            name=f"{commodity_chosen} Price",
            line={"color": "#1f2937"},
        )
    )

    buys = df_trades[df_trades["position"] == "buy"]
    sells = df_trades[df_trades["position"] == "sell"]

    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["trade_price"],
            mode="markers",
            name="Buy",
            marker={
                "symbol": "triangle-up",
                "size": 12,
                "color": "#16a34a",
                "line": {"color": "white", "width": 1},
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["trade_price"],
            mode="markers",
            name="Sell",
            marker={
                "symbol": "triangle-down",
                "size": 12,
                "color": "#dc2626",
                "line": {"color": "white", "width": 1},
            },
        )
    )

    if mtm_trade:
        mtm_ts = pd.Timestamp(str(mtm_trade["date"]))
        if mtm_ts in plot_df.index:
            fig.add_trace(
                go.Scatter(
                    x=[mtm_ts],
                    y=[plot_df[commodity_chosen].loc[mtm_ts]],
                    mode="markers",
                    name="Open MTM",
                    marker={
                        "symbol": "circle",
                        "size": 14,
                        "color": "#f59e0b",
                        "line": {"color": "white", "width": 2},
                    },
                )
            )

    fig.update_layout(
        title="Price & Trade Signals",
        xaxis_title="Date",
        yaxis_title=f"{commodity_chosen} (USD)",
        template="plotly_white",
    )
    return fig


def _ratio_chart(plot_df: pd.DataFrame, down_entry: float, up_exit: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["ratio"],
            name="Spread Ratio",
            line={"color": "#7c3aed"},
        )
    )
    fig.add_hline(
        y=down_entry,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text="Entry",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=up_exit,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Exit",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Ratio Behaviour (metric-ton terms)",
        xaxis_title="Date",
        yaxis_title="Ratio",
        template="plotly_white",
    )
    return fig


def _pnl_chart(df_trades: pd.DataFrame, mtm_trade: dict[str, object] | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_trades.index,
            y=df_trades["pnl_usd_cumsum"],
            mode="lines",
            name="Realised PnL",
            fill="tozeroy",
            line={"color": "#2563eb"},
        )
    )
    if mtm_trade:
        mtm_point = float(df_trades["pnl_usd_cumsum"].iloc[-1]) + float(str(mtm_trade["pnl_usd"]))
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(str(mtm_trade["date"]))],
                y=[mtm_point],
                mode="markers",
                name="MTM Adjustment",
                marker={"color": "#f59e0b", "size": 14, "line": {"color": "white", "width": 2}},
            )
        )
    fig.update_layout(
        title="Cumulative PnL (USD)",
        xaxis_title="Date",
        yaxis_title="PnL (USD)",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _streamlit_render(fig: go.Figure) -> None:
    if st is not None:
        st.plotly_chart(fig, use_container_width=True)


def _standalone_render(fig: go.Figure) -> None:
    fig.show()
