"""Plotly figure builders for the crush margin page. Pure, no UI imports."""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go

from .seasonal import REFERENCE_YEAR, to_reference_year

__all__ = [
    "feedstock_figure",
    "gap_figure",
    "history_figure",
    "oil_share_figure",
    "seasonal_figure",
    "value_figure",
]

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

SURFACE: Final[str] = "#141b23"
INK_SERIES: Final[str] = "#eef2f6"
MUTED: Final[str] = "#a3afbd"
GRID: Final[str] = "#232c37"
AXIS: Final[str] = "#3b4653"
# Amber is the crush margin and nothing else.
CRUSH: Final[str] = "#c58512"
OIL: Final[str] = "#4d8fe8"
MEAL: Final[str] = "#c76fae"
FLAT: Final[str] = "#7d8796"
BAND_FILL: Final[str] = "rgba(125, 135, 150, 0.22)"

MAIN_HEIGHT: Final[int] = 380
SMALL_HEIGHT: Final[int] = 200


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def seasonal_figure(
    values: pd.Series, flagged: pd.Series, band: pd.DataFrame, year: int
) -> go.Figure:
    """This year's crush over the monthly range of the five previous years."""
    fig = go.Figure()
    edges = [pd.Timestamp(REFERENCE_YEAR, month, 1) for month in range(1, 13)]
    edges.append(pd.Timestamp(REFERENCE_YEAR, 12, 31))
    low = [*band["low"], band["low"].iloc[-1]]
    high = [*band["high"], band["high"].iloc[-1]]
    median = [*band["median"], band["median"].iloc[-1]]
    previous = f"{year - 5} to {year - 1}"
    fig.add_trace(_step(edges, low, "Range low", FLAT, 0, hover=False, legend=False))
    fig.add_trace(_step(edges, high, f"Range {previous}", FLAT, 0, hover=False, fill="tonexty"))
    fig.add_trace(_step(edges, median, f"Median {previous}", FLAT, 1.25, dash="dot"))

    this_year = values[pd.DatetimeIndex(values.index).year == year]
    flags = flagged.reindex(this_year.index, fill_value=True)
    x = to_reference_year(pd.DatetimeIndex(this_year.index))
    fig.add_trace(_line(x, this_year, "Roll or delivery", FLAT, 1, hover=False))
    fig.add_trace(_line(x, this_year.where(~flags), str(year), CRUSH, 2.25))
    clean = this_year[~flags]
    if not clean.empty:
        fig.add_trace(_dot(to_reference_year(pd.DatetimeIndex(clean.index[-1:])), clean.iloc[-1:]))

    fig.update_layout(**_layout(MAIN_HEIGHT))
    fig.update_xaxes(
        range=[pd.Timestamp(REFERENCE_YEAR, 1, 1), pd.Timestamp(REFERENCE_YEAR, 12, 31)],
        tickformat="%b",
        dtick="M1",
        hoverformat="%d %b",
    )
    fig.update_yaxes(title="USD per tonne of soybeans", zeroline=True, zerolinecolor=AXIS)
    return fig


def history_figure(
    values: pd.Series,
    flagged: pd.Series,
    title: str,
    color: str = CRUSH,
    name: str = "Clean sessions",
    height: int = MAIN_HEIGHT,
) -> go.Figure:
    """A series over time, with roll and delivery sessions drawn as a thin grey line."""
    flags = flagged.reindex(values.index, fill_value=True)
    fig = go.Figure()
    fig.add_trace(_line(values.index, values, "Roll or delivery", FLAT, 1, hover=False))
    fig.add_trace(_line(values.index, values.where(~flags), name, color, 1.75))
    clean = values[~flags]
    if not clean.empty:
        fig.add_trace(_dot(clean.index[-1:], clean.iloc[-1:], color))
    fig.update_layout(**_layout(height))
    fig.update_xaxes(range=[values.index[0], values.index[-1]])
    fig.update_yaxes(title=title, zeroline=True, zerolinecolor=AXIS)
    return fig


def gap_figure(table: pd.DataFrame, height: int = 260) -> go.Figure:
    """Oil cost minus heating oil, before and after the D4 RIN credit, on clean sessions."""
    flags = table["flagged"].reindex(table.index, fill_value=True).astype(bool)
    fig = go.Figure()
    fig.add_trace(_line(table.index, table["gap"], "Roll or delivery", FLAT, 1, hover=False))
    fig.add_trace(_line(table.index, table["gap"].where(~flags), "Before the RIN", INK_SERIES, 1.5))
    if table["after_rin"].notna().any():
        after = table["after_rin"].where(~flags)
        fig.add_trace(_line(table.index, after, "After the D4 RIN", OIL, 1.75))
    fig.update_layout(**_layout(height))
    fig.update_xaxes(range=[table.index[0], table.index[-1]])
    fig.update_yaxes(title="Oil minus heating oil, USD per m³", zeroline=True, zerolinecolor=AXIS)
    return fig


def value_figure(table: pd.DataFrame) -> go.Figure:
    """Meal value, oil value and soybean cost, per tonne of soybeans crushed."""
    fig = go.Figure()
    fig.add_trace(_line(table.index, table["beans"], "Soybean cost", INK_SERIES, 1.5))
    fig.add_trace(_line(table.index, table["meal_value"], "Meal value", MEAL, 1.5))
    fig.add_trace(_line(table.index, table["oil_value"], "Oil value", OIL, 1.5))
    fig.update_layout(**_layout(MAIN_HEIGHT))
    fig.update_xaxes(range=[table.index[0], table.index[-1]])
    fig.update_yaxes(title="USD per tonne of soybeans")
    return fig


def oil_share_figure(table: pd.DataFrame) -> go.Figure:
    """Oil's share of the product value, with roll and delivery sessions in grey."""
    fig = history_figure(
        table["oil_share"] * 100,
        table["flagged"],
        "Oil share, %",
        color=OIL,
        name="Oil share",
        height=SMALL_HEIGHT,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(zeroline=False, ticksuffix="%", tickformat=".0f")
    return fig


def feedstock_figure(table: pd.DataFrame) -> go.Figure:
    """Soybean oil cost per m3 of biodiesel against heating oil per m3."""
    fig = go.Figure()
    fig.add_trace(_line(table.index, table["diesel"], "Heating oil", FLAT, 1.5))
    fig.add_trace(_line(table.index, table["feedstock"], "Soybean oil feedstock", OIL, 1.75))
    fig.update_layout(**_layout(MAIN_HEIGHT))
    fig.update_xaxes(range=[table.index[0], table.index[-1]])
    fig.update_yaxes(title="USD per m³")
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
            "tickformat": ",.0f",
            "ticklabelstandoff": 8,
            "title_font": {"size": 11},
        },
    }


def _line(x: Any, y: Any, name: str, color: str, width: float, hover: bool = True) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        hovertemplate="%{y:,.1f}" if hover else None,
        hoverinfo=None if hover else "skip",
        line={"color": color, "width": width},
    )


def _step(
    x: list[pd.Timestamp],
    y: list[float],
    name: str,
    color: str,
    width: float,
    dash: str | None = None,
    hover: bool = True,
    legend: bool = True,
    fill: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        showlegend=legend,
        fill=fill,
        fillcolor=BAND_FILL if fill else None,
        hovertemplate="%{y:,.1f}" if hover else None,
        hoverinfo=None if hover else "skip",
        line={"color": color, "width": width, "shape": "hv", "dash": dash},
    )


def _dot(x: Any, y: pd.Series, color: str = CRUSH) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=[float(value) for value in y],
        mode="markers",
        name="Latest clean session",
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
        marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
    )
