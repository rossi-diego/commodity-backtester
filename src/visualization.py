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

SURFACE: Final[str] = "#141b23"
INK_SERIES: Final[str] = "#eef2f6"
MUTED: Final[str] = "#a3afbd"
GRID: Final[str] = "#232c37"
AXIS: Final[str] = "#3b4653"
ENTRY: Final[str] = "#c58512"
EXIT: Final[str] = "#4d8fe8"
FLAT: Final[str] = "#7d8796"
LONG_BAND: Final[str] = "#c58512"
LONG_BAND_OPACITY: Final[float] = 0.1
REFERENCE: Final[str] = "#7d8796"

BACKTEST_HEIGHT: Final[int] = 420
RATIO_HEIGHT: Final[int] = 300
ZSCORE_HEIGHT: Final[int] = 180
# Fixed so the two stacked monitor charts align. With automargin the plot width
# depends on the width of the tick labels, and the shared dates drift apart.
MONITOR_RIGHT_MARGIN: Final[int] = 72

# Above this many fills the markers cover the price, so they start hidden.
MAX_VISIBLE_MARKERS: Final[int] = 60


def markers_hidden(trades: pd.DataFrame) -> bool:
    """Return True when the fills are too dense to draw by default."""
    return len(trades) > MAX_VISIBLE_MARKERS


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def price_signals_figure(window: pd.Series, trades: pd.DataFrame, unit: str) -> go.Figure:
    """Price line coloured by position, long periods as bands, fills as markers."""
    fig = go.Figure()
    if trades.empty:
        fig.add_trace(_price_line(window, str(window.name), FLAT, 1.5))
        last_color = FLAT
    else:
        state = _long_state(window, trades)
        previous = state.shift(1, fill_value=False)
        # Both segments include the session where the position changes, so the line stays joined.
        fig.add_trace(_price_line(window.where(state | previous), "Long", ENTRY, 2))
        fig.add_trace(_price_line(window.where(~state | ~previous), "Flat", FLAT, 1.5))
        visible: bool | str = "legendonly" if markers_hidden(trades) else True
        fig.add_trace(_fill_markers(trades, "buy", "Entry", "triangle-up", ENTRY, visible))
        fig.add_trace(_fill_markers(trades, "sell", "Exit", "triangle-down", EXIT, visible))
        last_color = ENTRY if bool(state.iloc[-1]) else FLAT
    fig.add_trace(_last_dot(window, last_color, "Last close"))

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
            line={"color": INK_SERIES, "width": 1.75, "shape": "hv"},
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
                line={"color": ENTRY, "width": 1.75, "dash": "dot"},
                marker={"color": ENTRY, "size": [0, 9], "line": {"color": SURFACE, "width": 2}},
            )
        )

    fig.update_layout(**_layout(BACKTEST_HEIGHT))
    # The zero line stays because the sign of this series is the point of the warning.
    fig.update_yaxes(
        title="USD, per contract, unadjusted", zeroline=True, zerolinecolor=AXIS, zerolinewidth=1
    )
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
            line={"color": ENTRY, "width": 1.5, "dash": "dot"},
        )
    )
    fig.add_trace(_last_dot(ratio, INK_SERIES, "Last"))
    fig.update_layout(**_layout(RATIO_HEIGHT))
    fig.update_layout(margin_r=MONITOR_RIGHT_MARGIN)
    fig.update_xaxes(range=_date_range(ratio))
    fig.update_yaxes(title="Ratio, USD/MT basis", automargin=False, tickformat=".1f")
    return fig


def zscore_figure(stats: pd.DataFrame) -> go.Figure:
    """Trailing 1-year z-score with reference lines at +/-2. Same x-axis as the ratio."""
    fig = go.Figure()
    z = stats["z"]
    fig.add_trace(_line(stats.index, z, "Z-score", FLAT))
    # Neighbours are included so each stretch beyond +/-2 draws as a closed segment.
    beyond = (z.abs() > 2) | (z.abs().shift(1) > 2) | (z.abs().shift(-1) > 2)
    fig.add_trace(
        go.Scatter(
            x=stats.index,
            y=z.where(beyond),
            mode="lines",
            name="Beyond +/-2",
            hoverinfo="skip",
            line={"color": INK_SERIES, "width": 1.75},
        )
    )
    fig.add_trace(_last_dot(z, INK_SERIES, "Last"))
    for level in (-2, 2):
        fig.add_hline(y=level, line={"color": REFERENCE, "width": 1, "dash": "dot"})
    fig.update_layout(**_layout(ZSCORE_HEIGHT), showlegend=False)
    fig.update_layout(margin_r=MONITOR_RIGHT_MARGIN)
    fig.update_xaxes(range=_date_range(stats))
    fig.update_yaxes(
        title="Z-score (1y)", automargin=False, tickformat="+d", zeroline=True, zerolinecolor=AXIS
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout(height: int) -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        "font": {"family": "IBM Plex Sans, sans-serif", "size": 12, "color": MUTED},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.02, "x": 0, "xanchor": "left", "yanchor": "bottom"},
        "xaxis": {"linecolor": AXIS, "showgrid": False, "ticks": "outside", "tickcolor": AXIS},
        "yaxis": {
            "gridcolor": GRID,
            "zeroline": False,
            "side": "right",
            "tickformat": ",",
            "ticklabelstandoff": 8,
            "title_font": {"size": 11},
        },
    }


def _line(x: Any, y: Any, name: str, color: str = INK_SERIES) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": 1.5},
    )


def _price_line(y: pd.Series, name: str, color: str, width: float) -> go.Scatter:
    return go.Scatter(
        x=y.index,
        y=y,
        mode="lines",
        name=name,
        hovertemplate="%{y:,.2f}",
        line={"color": color, "width": width},
    )


def _last_dot(series: pd.Series, color: str, name: str) -> go.Scatter:
    return go.Scatter(
        x=[series.index[-1]],
        y=[float(series.iloc[-1])],
        mode="markers",
        name=name,
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
        marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
    )


def _long_state(window: pd.Series, trades: pd.DataFrame) -> pd.Series:
    """Return, for each session of the window, whether the rule holds the contract."""
    is_buy = (trades["position"] == "buy").groupby(level=0).last()
    return is_buy.reindex(window.index, method="ffill").fillna(False).astype(bool)


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
            "line": {"color": SURFACE, "width": 1},
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
