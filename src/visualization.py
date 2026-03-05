"""Plotly chart builders for the backtest results."""

from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st  # type: ignore[import-untyped]

    _HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False


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
) -> None:
    """Render three interactive Plotly charts.

    Charts
    ------
    1. **Price + trade signals** — price line with ▲ buy / ▼ sell markers.
    2. **Ratio behaviour** — spread ratio with entry/exit threshold lines.
    3. **Cumulative PnL** — realised equity curve with optional MTM dot.

    Parameters
    ----------
    df_prices:
        Full price DataFrame (must contain a ``"ratio"`` column when this
        function is called — populated by the strategy engine).
    df_trades:
        Trade log output of :func:`utils.pnl_trades`.
    commodity_chosen:
        Primary commodity display name.
    down_entry / up_exit:
        Entry and exit ratio thresholds (shown as horizontal reference lines).
    start_date / end_date:
        Date range to display.
    mtm_trade:
        Open-position mark-to-market point, or ``None``.
    tons_conversion:
        Unused here; kept for API compatibility.
    use_streamlit:
        If ``True``, renders via ``st.plotly_chart``; otherwise calls
        ``fig.show()``.
    """
    plot_df = df_prices.copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    plot_df = plot_df.loc[str(start_date) : str(end_date)]

    _render = _streamlit_render if (use_streamlit and _HAS_STREAMLIT) else _standalone_render

    # ── 1. Price + signals ────────────────────────────────────────────────────
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df[commodity_chosen],
        mode="lines",
        name=f"{commodity_chosen} Price",
        line={"color": "#1f2937"},
    ))

    buys  = df_trades[df_trades["position"] == "buy"]
    sells = df_trades[df_trades["position"] == "sell"]

    fig_price.add_trace(go.Scatter(
        x=buys.index, y=buys["trade_price"],
        mode="markers", name="Buy",
        marker={"symbol": "triangle-up", "size": 12, "color": "#16a34a",
                "line": {"color": "white", "width": 1}},
    ))
    fig_price.add_trace(go.Scatter(
        x=sells.index, y=sells["trade_price"],
        mode="markers", name="Sell",
        marker={"symbol": "triangle-down", "size": 12, "color": "#dc2626",
                "line": {"color": "white", "width": 1}},
    ))

    if mtm_trade:
        mtm_ts = pd.Timestamp(str(mtm_trade["date"]))
        if mtm_ts in plot_df.index:
            fig_price.add_trace(go.Scatter(
                x=[mtm_ts],
                y=[plot_df[commodity_chosen].loc[mtm_ts]],
                mode="markers", name="Open MTM",
                marker={"symbol": "circle", "size": 14, "color": "#f59e0b",
                        "line": {"color": "white", "width": 2}},
            ))

    fig_price.update_layout(
        title="Price & Trade Signals",
        xaxis_title="Date",
        yaxis_title=f"{commodity_chosen} (USD)",
        template="plotly_white",
    )
    _render(fig_price)

    # ── 2. Ratio ──────────────────────────────────────────────────────────────
    if "ratio" not in plot_df.columns:
        raise ValueError(
            "df_prices must contain a 'ratio' column. "
            "Ensure the strategy engine has been run before calling backtest_charts()."
        )

    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["ratio"],
        name="Spread Ratio",
        line={"color": "#7c3aed"},
    ))
    fig_ratio.add_hline(
        y=down_entry, line_dash="dash", line_color="#16a34a",
        annotation_text="Entry", annotation_position="bottom left",
    )
    fig_ratio.add_hline(
        y=up_exit, line_dash="dash", line_color="#dc2626",
        annotation_text="Exit", annotation_position="top left",
    )
    fig_ratio.update_layout(
        title="Ratio Behaviour (metric-ton terms)",
        xaxis_title="Date",
        yaxis_title="Ratio",
        template="plotly_white",
    )
    _render(fig_ratio)

    # ── 3. Cumulative PnL ─────────────────────────────────────────────────────
    if "pnl_usd_cumsum" not in df_trades.columns:
        return

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_trades.index,
        y=df_trades["pnl_usd_cumsum"],
        mode="lines",
        name="Realised PnL",
        fill="tozeroy",
        line={"color": "#2563eb"},
    ))

    if mtm_trade:
        mtm_point = float(df_trades["pnl_usd_cumsum"].iloc[-1]) + float(str(mtm_trade["pnl_usd"]))
        fig_pnl.add_trace(go.Scatter(
            x=[pd.Timestamp(str(mtm_trade["date"]))],
            y=[mtm_point],
            mode="markers", name="MTM Adjustment",
            marker={"color": "#f59e0b", "size": 14,
                    "line": {"color": "white", "width": 2}},
        ))

    fig_pnl.update_layout(
        title="Cumulative PnL (USD)",
        xaxis_title="Date",
        yaxis_title="PnL (USD)",
        template="plotly_white",
    )
    _render(fig_pnl)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _streamlit_render(fig: go.Figure) -> None:
    if st is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


def _standalone_render(fig: go.Figure) -> None:
    fig.show()
