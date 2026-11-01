"""Plotly figure builders for the backtest and the ratio monitor. Pure, no UI imports."""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "MAX_VISIBLE_MARKERS",
    "markers_hidden",
    "price_signals_figure",
    "ratio_figure",
    "unadjusted_pnl_figure",
    "zscore_figure",
]

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

INK_SERIES: Final[str] = "#3b4252"
MUTED: Final[str] = "#52514e"
GRID: Final[str] = "#eef0f3"
AXIS: Final[str] = "#d4d8de"
ENTRY: Final[str] = "#2a78d6"
EXIT: Final[str] = "#eb6834"
LONG_BAND: Final[str] = "#2a78d6"
LONG_BAND_OPACITY: Final[float] = 0.07
REFERENCE: Final[str] = "#9aa1ac"

BACKTEST_HEIGHT: Final[int] = 420
RATIO_HEIGHT: Final[int] = 300
ZSCORE_HEIGHT: Final[int] = 180
# Fixed so the two stacked monitor charts align. With automargin the plot width
# depends on the width of the tick labels, and the shared dates drift apart.
MONITOR_RIGHT_MARGIN: Final[int] = 64

# Above this many fills the markers cover the price, so they start hidden.
MAX_VISIBLE_MARKERS: Final[int] = 60


def markers_hidden(trades: pd.DataFrame) -> bool:
    """Return True when the fills are too dense to draw by default."""
    return len(trades) > MAX_VISIBLE_MARKERS


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def price_signals_figure(window: pd.Series, trades: pd.DataFrame, unit: str) -> go.Figure:
    """Price line with long periods as bands and fills as markers."""
    fig = go.Figure()
    fig.add_trace(_line(window.index, window, str(window.name)))

    if not trades.empty:
        visible: bool | str = "legendonly" if markers_hidden(trades) else True
        fig.add_trace(_fill_markers(trades, "buy", "Entry", "triangle-up", ENTRY, visible))
        fig.add_trace(_fill_markers(trades, "sell", "Exit", "triangle-down", EXIT, visible))

    # Bands go in one list. Calling add_vrect in a loop is quadratic in the band count.
    fig.update_layout(**_layout(BACKTEST_HEIGHT), shapes=_long_bands(trades, window))
    # Explicit range: autorange pads for the markers and leaves a blank strip at the start.
    fig.update_xaxes(range=_date_range(window))
    fig.update_yaxes(title=unit)
    return fig


def unadjusted_pnl_figure(
    trades: pd.DataFrame,
    window: pd.Series,
    mtm: dict[str, object] | None,
) -> go.Figure:
    """Realised P&L as a step line over the window, plus the open position if any."""
    start, end = window.index[0], window.index[-1]
    if trades.empty:
        dates, realised = [start, end], [0.0, 0.0]
    else:
        cumulative = [float(v) for v in trades["pnl_usd_cumsum"]]
        dates = [start, *trades.index, end]
        realised = [0.0, *cumulative, cumulative[-1]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=realised,
            mode="lines",
            name="Realised",
            line={"color": INK_SERIES, "width": 1.5, "shape": "hv"},
        )
    )

    if mtm:
        open_from = trades.index[-1]
        fig.add_trace(
            go.Scatter(
                x=[open_from, pd.Timestamp(str(mtm["date"]))],
                y=[realised[-1], realised[-1] + float(str(mtm["pnl_usd"]))],
                mode="lines+markers",
                name="With open position",
                line={"color": ENTRY, "width": 1.5, "dash": "dot"},
                marker={"color": ENTRY, "size": [0, 9], "line": {"color": "white", "width": 1}},
            )
        )

    fig.update_layout(**_layout(BACKTEST_HEIGHT))
    fig.update_yaxes(title="USD, per contract, unadjusted")
    return fig


def ratio_figure(ratio: pd.Series, stats: pd.DataFrame) -> go.Figure:
    """Ratio with its trailing 1-year mean. ``stats`` must share the ratio's window."""
    fig = go.Figure()
    fig.add_trace(_line(ratio.index, ratio, "Ratio"))
    fig.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["mean_1y"],
            mode="lines",
            name="1y mean",
            line={"color": ENTRY, "width": 1.25, "dash": "dot"},
        )
    )
    fig.update_layout(**_layout(RATIO_HEIGHT))
    fig.update_layout(margin_r=MONITOR_RIGHT_MARGIN)
    fig.update_xaxes(range=_date_range(ratio))
    fig.update_yaxes(title="Ratio, USD/MT basis", automargin=False)
    return fig


def zscore_figure(stats: pd.DataFrame) -> go.Figure:
    """Trailing 1-year z-score with reference lines at +/-2. Same x-axis as the ratio."""
    fig = go.Figure()
    fig.add_trace(_line(stats.index, stats["z"], "Z-score"))
    for level in (-2, 2):
        fig.add_hline(y=level, line={"color": REFERENCE, "width": 1, "dash": "dot"})
    fig.update_layout(**_layout(ZSCORE_HEIGHT), showlegend=False)
    fig.update_layout(margin_r=MONITOR_RIGHT_MARGIN)
    fig.update_xaxes(range=_date_range(stats))
    fig.update_yaxes(title="Z-score (1y)", automargin=False)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout(height: int) -> dict[str, Any]:
    return {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        "font": {"family": "Inter, sans-serif", "size": 12, "color": MUTED},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.04, "x": 0, "xanchor": "left", "yanchor": "bottom"},
        "xaxis": {"gridcolor": GRID, "linecolor": AXIS},
        "yaxis": {"gridcolor": GRID, "zeroline": False, "side": "right"},
    }


def _line(x: Any, y: Any, name: str) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": INK_SERIES, "width": 1.5},
    )


def _fill_markers(
    trades: pd.DataFrame,
    position: str,
    name: str,
    symbol: str,
    color: str,
    visible: bool | str,
) -> go.Scatter:
    fills = trades[trades["position"] == position]
    return go.Scatter(
        x=fills.index,
        y=fills["trade_price"],
        mode="markers",
        name=name,
        visible=visible,
        cliponaxis=False,
        marker={
            "symbol": symbol,
            "size": 9,
            "color": color,
            "line": {"color": "white", "width": 1},
        },
    )


def _long_bands(trades: pd.DataFrame, window: pd.Series) -> list[dict[str, Any]]:
    """Return one rectangle per long period, from the entry fill to the exit or window end."""
    if trades.empty:
        return []
    entries = trades.index[trades["position"] == "buy"]
    exits = trades.index[trades["position"] == "sell"]
    window_end = window.index[-1]
    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": entry,
            "x1": exits[i] if i < len(exits) else window_end,
            "y0": 0,
            "y1": 1,
            "fillcolor": LONG_BAND,
            "opacity": LONG_BAND_OPACITY,
            "line_width": 0,
            "layer": "below",
        }
        for i, entry in enumerate(entries)
    ]


def _date_range(data: pd.Series | pd.DataFrame) -> list[pd.Timestamp]:
    return [data.index[0], data.index[-1]]
