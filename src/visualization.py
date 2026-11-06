"""Plotly figure builders for the crush margin page. Pure, no UI imports.

Every section draws the same two figures, so the three charts read alike:
``history_figure`` (a metric over time) and ``by_month_figure`` (monthly
averages, one line per year).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .seasonal import monthly_band

__all__ = ["by_month_figure", "history_figure"]

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
# Past years in the by-month view, by how many years back: teal is last year. Hues kept
# clear of amber, blue and the near-white ink, which the current year draws in. Each year
# keeps its colour whichever others are picked; six to ten years back repeat the five
# colours, dashed.
YEAR_COLORS: Final[tuple[str, ...]] = ("#2bb39b", "#a98bf0", "#86b83a", "#e5664c", "#d86fb0")

HEIGHT: Final[int] = 400
MONTHS: Final[list[str]] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]  # fmt: skip


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def history_figure(
    values: pd.Series,
    flagged: pd.Series,
    *,
    name: str,
    color: str,
    y_title: str,
    percent: bool = False,
    reference: tuple[str, pd.Series] | None = None,
    held: tuple[str, pd.Series] | None = None,
    zero_labels: tuple[str, str] | None = None,
    hover: dict[str, pd.Series] | None = None,
) -> go.Figure:
    """A metric over time, with roll and delivery sessions drawn as a thin grey line.

    ``reference`` adds a thin grey comparison line. ``held`` marks the sessions whose
    value rests on a carried-forward input; they are drawn dashed and named.
    ``hover`` adds component levels to the tooltip of the main line.
    """
    flags = flagged.reindex(values.index, fill_value=True).astype(bool)
    held_mask = (
        held[1].reindex(values.index, fill_value=False).astype(bool)
        if held is not None
        else pd.Series(False, index=values.index)
    )
    fig = go.Figure()
    if reference is not None:
        ref_name, ref_values = reference
        fig.add_trace(_line(values.index, ref_values.where(~flags), ref_name, FLAT, 1, False))
    fig.add_trace(_line(values.index, values, "Roll or delivery", FLAT, 1, hover=False))
    main = _line(values.index, values.where(~flags & ~held_mask), name, color, 1.75)
    _add_hover(main, values.index, hover, percent)
    fig.add_trace(main)
    if held is not None and held_mask.any():
        # Join the dashed tail to the last real point so the line does not break.
        start = int(np.flatnonzero(held_mask.to_numpy())[0])
        tail = values.iloc[max(start - 1, 0) :].where(~flags)
        fig.add_trace(_line(tail.index, tail, held[0], color, 1.75, dash="dash"))
    clean = values[~flags]
    if not clean.empty:
        fig.add_trace(_dot(clean.index[-1:], clean.iloc[-1:], color))
    fig.update_layout(**_layout())
    fig.update_xaxes(range=[values.index[0], values.index[-1]])
    _y_axis(fig, y_title, percent)
    if zero_labels is not None:
        _zero_labels(fig, values, zero_labels)
    return fig


def by_month_figure(
    means: pd.DataFrame,
    counts: pd.DataFrame,
    *,
    years: list[int],
    current: int,
    color: str,
    y_title: str,
    band: bool = True,
    percent: bool = False,
    month_to_date: int | None = None,
) -> go.Figure:
    """Monthly averages of clean sessions, one line per year, on a January-to-December axis.

    ``current`` is drawn bold in the metric colour; ``years`` are earlier years, each
    in its own colour. ``month_to_date`` is the month in progress, drawn as a hollow marker
    and not joined to the line. ``band`` shades the range of the five years before
    ``current``, with their average dotted.
    """
    fmt = _value_format(percent)
    fig = go.Figure()
    if band:
        stats = monthly_band(means, current)
        previous = f"{current - 5}–{current - 1}"
        fig.add_trace(_band_edge(stats["low"], "Range low", legend=False))
        fig.add_trace(_band_edge(stats["high"], f"Range of {previous} averages", fill=True))
        fig.add_trace(_month_line(stats["average"], f"Average {previous}", FLAT, 1.25, fmt, "dot"))
    # Oldest first, so the reversed legend reads from the current year back.
    past = sorted(year for year in years if year != current and year in means)
    for year in past:
        back, cycle = divmod(current - year - 1, len(YEAR_COLORS))
        fig.add_trace(
            _month_line(
                means[year],
                str(year),
                YEAR_COLORS[cycle],
                1.5,
                fmt,
                "dash" if back else None,
                counts=counts[year],
            )
        )
    if current in means:
        line = means[current]
        if month_to_date is not None:
            line = line.where(line.index != month_to_date)
        fig.add_trace(_month_line(line, str(current), color, 2.5, fmt, counts=counts[current]))
        to_date = None if month_to_date is None else means[current].iloc[month_to_date - 1]
        if month_to_date is not None and pd.notna(to_date):
            fig.add_trace(
                go.Scatter(
                    x=[MONTHS[month_to_date - 1]],
                    y=[float(to_date)],
                    mode="markers",
                    name=f"{MONTHS[month_to_date - 1]} to date",
                    customdata=[int(counts[current].iloc[month_to_date - 1])],
                    hovertemplate=fmt + " (%{customdata} sessions so far)",
                    marker={"size": 10, "color": SURFACE, "line": {"color": color, "width": 2}},
                    cliponaxis=False,
                )
            )
    fig.update_layout(**_layout())
    fig.update_layout(legend={"traceorder": "reversed"})
    fig.update_xaxes(categoryorder="array", categoryarray=MONTHS, range=[-0.3, 11.3])
    _y_axis(fig, y_title, percent)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout() -> dict[str, Any]:
    return {
        "template": "plotly_dark",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "height": HEIGHT,
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


def _y_axis(fig: go.Figure, title: str, percent: bool) -> None:
    if percent:
        fig.update_yaxes(title=title, ticksuffix="%", tickformat=".0f")
    else:
        fig.update_yaxes(title=title, zeroline=True, zerolinecolor=AXIS)


def _value_format(percent: bool) -> str:
    return "%{y:.1f}%" if percent else "%{y:,.1f}"


def _line(
    x: Any,
    y: Any,
    name: str,
    color: str,
    width: float,
    hover: bool = True,
    dash: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        hovertemplate="%{y:,.1f}" if hover else None,
        hoverinfo=None if hover else "skip",
        line={"color": color, "width": width, "dash": dash},
    )


def _add_hover(
    trace: go.Scatter, index: pd.Index, hover: dict[str, pd.Series] | None, percent: bool
) -> None:
    template = _value_format(percent)
    if hover:
        names = list(hover)
        data = pd.concat([hover[name].reindex(index) for name in names], axis=1)
        trace.update(customdata=data.to_numpy())
        template += "".join(
            f"<br>{name}: %{{customdata[{i}]:,.0f}}" for i, name in enumerate(names)
        )
    trace.update(hovertemplate=template)


def _month_line(
    values: pd.Series,
    name: str,
    color: str,
    width: float,
    fmt: str,
    dash: str | None = None,
    counts: pd.Series | None = None,
) -> go.Scatter:
    trace = go.Scatter(
        x=MONTHS,
        y=values.to_numpy(),
        mode="lines+markers" if dash is None else "lines",
        name=name,
        connectgaps=False,
        hovertemplate=fmt,
        line={"color": color, "width": width, "dash": dash},
        marker={"size": 5, "color": color},
    )
    if counts is not None:
        trace.update(customdata=counts.to_numpy(), hovertemplate=fmt + " (%{customdata} sessions)")
    return trace


def _band_edge(values: pd.Series, name: str, legend: bool = True, fill: bool = False) -> go.Scatter:
    return go.Scatter(
        x=MONTHS,
        y=values.to_numpy(),
        mode="lines",
        name=name,
        showlegend=legend,
        fill="tonexty" if fill else None,
        fillcolor=BAND_FILL if fill else None,
        hoverinfo="skip",
        line={"color": FLAT, "width": 0},
    )


def _zero_labels(fig: go.Figure, values: pd.Series, labels: tuple[str, str]) -> None:
    """Name both sides of zero, when zero is inside the drawn range."""
    if values.dropna().empty or not values.min() < 0 < values.max():
        return
    for text, shift in ((labels[0], 10), (labels[1], -10)):
        fig.add_annotation(
            x=0.005,
            xref="paper",
            y=0,
            yshift=shift,
            text=text,
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": MUTED},
        )


def _dot(x: Any, y: pd.Series, color: str) -> go.Scatter:
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
